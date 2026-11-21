# Deploying Triage

| Piece | Where | Config in this repo |
|---|---|---|
| API (FastAPI) | One Oracle Cloud **Always Free** VM: Ampere A1 (ARM), Ubuntu, Docker | `Dockerfile`, `docker-compose.yml`, `deploy/` |
| HTTPS | Caddy on the same VM, automatic Let's Encrypt certificates for `api.gettriage.dev` | `deploy/Caddyfile` |
| Database | Neon Postgres (pooled connection for the app, direct for migrations) | `migrations/` (Alembic) |
| Website (React) | Vercel, root directory `web/` | `web/vercel.json` |
| Email | Resend | environment variables only |

Never put real values in this file or anywhere in the repo. Secrets live in `.env` on the server (git-ignored) and
in the Vercel dashboard.

---

## 1. Oracle Cloud: stay inside Always Free

The Always Free allowance for Ampere A1 is **1,500 OCPU hours and 9,000 GB hours per month**, which for an Always
Free tenancy equals 2 OCPUs and 12 GB of memory running all month
([Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)).
The allowance is shared by **every A1 instance in the tenancy**.

> **Going over the A1 allowance disables ALL A1 instances in the tenancy**, not just the newest one, and Oracle
> deletes them after 30 days unless the account is upgraded
> ([Free Tier FAQ](https://www.oracle.com/cloud/free/faq/)). Since the 2026 reduction, users have also reported
> instances staying disabled after getting back under the limit. Don't create a second A1 instance, and don't
> resize beyond the allowance.

**Use 1 OCPU and 3 GB of memory.** That's 1 × 730 ≈ 730 OCPU hours and 3 × 730 ≈ 2,190 GB hours a month, well
inside the allowance, and it fits Triage comfortably (section 7).

**Idle instances can be reclaimed.** Oracle may reclaim an Always Free instance that, over 7 days, has 95th-percentile
CPU below 20%, network below 20% **and** memory below 20% (the memory rule applies to A1 shapes only)
([Always Free Resources](https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm)).
A quiet API will be under 20% CPU and network, so memory is what keeps it from looking idle. On 3 GB, the 20% line is
about 615 MB; the running stack uses more than that on its own (section 7). This is why the VM is sized at 3 GB and not
12 GB: on 12 GB the same usage would be under 20%. If the API stops (crashes and stays down), memory drops and the VM
can look idle: check it after any outage.

Other Always Free limits that matter here: 200 GB of block storage in total (the boot volume defaults to 50 GB, which
is plenty) and 10 TB of outbound data a month.

## 2. Create and secure the VM

1. **Create the instance:** Compute > Instances > Create. Image: **Canonical Ubuntu** (latest LTS, aarch64). Shape:
   **VM.Standard.A1.Flex**, **1 OCPU, 3 GB**. It's marked "Always Free-eligible". Use your **home region** (Always Free
   resources must be there).
2. **SSH key:** upload your **public** key (or let Oracle generate a pair and download the private key). There is no
   password login.
3. **Public IP:** the instance gets a public IP; DNS will point at it. If you might ever recreate the instance, use
   a **reserved** public IP instead (VNIC > IPv4 addresses), so the address and DNS don't change; check that it's
   within your tenancy's free limits first.
4. **Open ports 80 and 443 in the VCN.** Networking > Virtual cloud networks > your VCN > Security Lists > Default
   Security List > Add Ingress Rules, source `0.0.0.0/0`:
   - TCP, destination port 80
   - TCP, destination port 443
   - UDP, destination port 443 (HTTP/3)

   Keep the existing rule for TCP 22 (SSH). If you can, limit port 22's source to your own IP.
5. **Open the same ports in Ubuntu's firewall.** Oracle's Ubuntu images ship iptables rules that reject everything
   except SSH, so the VCN rule alone isn't enough. On the VM (`ssh ubuntu@<public-ip>`), add the rules at the top of
   the INPUT chain, before Oracle's REJECT rule, and save them so they survive a reboot:

   ```bash
   sudo iptables -I INPUT 1 -p tcp -m multiport --dports 80,443 -m conntrack --ctstate NEW -j ACCEPT
   sudo iptables -I INPUT 1 -p udp --dport 443 -m conntrack --ctstate NEW -j ACCEPT
   sudo netfilter-persistent save
   sudo iptables -L INPUT -n --line-numbers | head    # the ACCEPT rules come before the REJECT line
   ```

   Oracle advises against enabling `ufw` on its Ubuntu images; edit iptables as above. Docker adds its own rules for
   the containers' ports when it starts.
6. **SSH: keys only.** Oracle's Ubuntu images already disable passwords; make it explicit, and turn off root login:

   ```bash
   printf 'PasswordAuthentication no\nKbdInteractiveAuthentication no\nPermitRootLogin no\n' \
     | sudo tee /etc/ssh/sshd_config.d/99-triage.conf
   sudo sshd -t && sudo systemctl reload ssh
   sudo sshd -T | grep -E '^(passwordauthentication|kbdinteractiveauthentication|permitrootlogin) '
   ```

   Keep your current SSH session open and check that a new one still logs in before closing it.
7. **Security updates, automatically:**

   ```bash
   sudo apt update && sudo apt install -y unattended-upgrades
   sudo dpkg-reconfigure -plow unattended-upgrades          # answer Yes
   cat /etc/apt/apt.conf.d/20auto-upgrades                   # both lines should be "1"
   ```

   It installs security updates daily. Kernel updates need a reboot: check for `/var/run/reboot-required` now and
   then, and reboot in a quiet moment (the stack restarts by itself).
8. **Install Docker** (Engine and the Compose plugin, from Docker's own repository):

   ```bash
   sudo install -m 0755 -d /etc/apt/keyrings
   sudo curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
   echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
     | sudo tee /etc/apt/sources.list.d/docker.list
   sudo apt update && sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
   sudo usermod -aG docker ubuntu      # then log out and back in
   docker run --rm hello-world
   ```

## 3. Neon (database)

1. Create a project in the AWS region closest to your Oracle home region (for example AWS US West 2, Oregon, for
   Oracle's US West regions).
2. From **Connect**, copy two connection strings for the same database and role:
   - **Pooled** (host contains `-pooler`): this is `DATABASE_URL`.
   - **Direct** (no `-pooler`): this is `DATABASE_URL_UNPOOLED`, used only for migrations.
3. Change the scheme from `postgresql://` to `postgresql+psycopg://` in both, and keep `?sslmode=require`.

**Prepared statements are off automatically on the pooled host.** Neon's pooler (PgBouncer, transaction mode) can
send each transaction to a different server connection, where psycopg's prepared statements don't exist. Triage
turns them off when the host contains `-pooler` (`TRIAGE_DB_PREPARED_STATEMENTS=auto`, the default; `on` or `off`
overrides it). Migrations run on the direct connection, as Neon recommends.

### Stay inside the Free plan: let the database sleep

**Budget:** the Free plan gives **100 CU-hours per project per month**, which is about 400 hours of a 0.25 CU compute.
A month is about 730 hours, so the database **must** spend most of its time scaled to zero. If the CU-hours run out,
Neon suspends the compute until the next billing period (or an upgrade). The API stays up (`/healthz` is 200) but
everything that needs the database fails: the demo, sign-in, accounts and the request log. Production is its own
Neon project, so the test project's usage doesn't count against it.

**Set the production compute to a fixed 0.25 CU** (minimum 0.25, maximum 0.25): in the Neon console, edit the
production branch's compute and set its size. Free defaults to autoscaling from 0.25 up to 2 CU, which can burn
the budget up to 8 times as fast. Changing the size restarts the compute and drops open connections for a moment.

**Scale to zero:** Neon suspends the compute after **5 minutes** with no activity, and this can't be changed on
Free. Any query or new connection wakes it, and each wake is billed for at least those 5 minutes (about 0.02 CU-hours
at 0.25 CU). What touches the database:

| What | How often | Wakes Neon? |
|---|---|---|
| Real requests: demo chat, sign-in, account pages, API calls | When people use the site | Yes: that's the point |
| `GET /readyz` | Only when you run it | Yes: never point a monitor at it |
| Request-log retention (12 months) | At startup, then once a day | Yes: about 2.5 hours awake a month, about 0.6 CU-hours |
| `alembic upgrade head` and the schema check | Each deploy or restart | Yes, once |
| `GET /healthz`, Docker's healthcheck, uptime monitors | Every 30 seconds | **No**: never touches the database |
| Response-cache purge | Hourly | **No**: a local file |
| Open but idle pooled connections | Between requests | **No** (measured below) |

Measured on 2026-11-21 on a Free-plan test project: with one idle connection held open, pooled and then direct, the
compute still suspended after 5 minutes, and Neon dropped the held connection. The app's pool notices a dropped
connection before using it (pre-ping) and opens a new one.

**The first request after a sleep** waits for Neon to wake. Neon puts the wake at a few hundred milliseconds; in
the same test a first connection plus query took 1.1 to 1.3 seconds, against about 0.5 seconds for an awake
database. Postgres connections get a 10-second connect timeout and one retry after a second, so that request is
slower rather than failed. `/readyz` gives the database only 2 seconds, so after a sleep it can occasionally
return 503: check again.

**Restores:** the Free plan's history window is **6 hours** (capped at 1 GB of changes): within it you can restore
the branch to any moment (instant restore, also called point-in-time restore). Older than that, there is no
backup. Leave the window at its default. The privacy policy says backups may hold deleted data for **up to 6
hours**, which matches; if you move to a plan with a longer window, update that line too.

Check usage now and then in the Neon console's usage page, especially after launch or a traffic spike.

## 4. Environment variables and secrets

### The server's `.env`

On the VM, in the repo folder: `cp deploy/.env.example .env`, then fill it in (`nano .env`). It's git-ignored and
only lives on the server. `chmod 600 .env` so only your user can read it.

| Name | Purpose | Example format | Notes |
|---|---|---|---|
| `TRIAGE_IMAGE_TAG`, `TRIAGE_PREVIOUS_IMAGE_TAG` | The running image and the one kept for rollback | written by the scripts | Leave empty |
| `API_DOMAIN` | The API's domain, for Caddy's certificate | `api.gettriage.dev` | Empty: `api.gettriage.dev` |
| `ACME_EMAIL` | Optional Let's Encrypt contact, for certificate expiry warnings | `you@example.com` | Empty: none registered |
| `DATABASE_URL` | The app's database: Neon **pooled** string | `postgresql+psycopg://USER:PASSWORD@ep-xxx-pooler.REGION.aws.neon.tech/neondb?sslmode=require` | Secret |
| `DATABASE_URL_UNPOOLED` | Migrations only: Neon **direct** string | `postgresql+psycopg://USER:PASSWORD@ep-xxx.REGION.aws.neon.tech/neondb?sslmode=require` | Secret |
| `TRIAGE_DB_PREPARED_STATEMENTS` | Prepared statements | `auto` (default), `on`, `off` | Leave empty |
| `TRIAGE_HASH_SECRET` | Keys prompt fingerprints, cache keys, visitor ids and account API key hashes | 64 hex characters | **Generate fresh.** Changing it later invalidates every account API key |
| `TRIAGE_ENCRYPTION_KEYS` | Encrypts saved provider keys (AES-256-GCM) | `k1:<base64 of 32 bytes>` | **Generate fresh.** Required with accounts on. Rotation: `k2:<new>,k1:<old>`, then `scripts/rotate_provider_keys.py` |
| `TRIAGE_API_KEYS` | Bearer keys for your own apps | `key1,key2` | **Generate fresh.** Optional |
| `TRIAGE_ADMIN_KEYS` | Keys that may read and change the live routing config | `key1` | **Generate fresh.** Optional; keep separate from API keys |
| `GROQ_API_KEY` | The server's Groq key (demo and server-key requests) | `gsk_...` | Secret. A production key, not your development one |
| `GEMINI_API_KEY` | The server's Gemini key | `AIza...` | Secret |
| `TRIAGE_PUBLIC` | The free demo and bring-your-own-key chat without a Triage key | `on` | |
| `TRIAGE_ACCOUNTS` | User accounts | `on` | Needs `TRIAGE_ENCRYPTION_KEYS` |
| `TRIAGE_CORS_ORIGINS` | Browser origins allowed to call the API | `https://gettriage.dev,https://www.gettriage.dev` | Exactly the website's origins, no trailing slash |
| `TRIAGE_APP_URL` | The website, for links in emails | `https://gettriage.dev` | https only |
| `TRIAGE_PUBLIC_API_URL` | The API address shown in the welcome email | `https://api.gettriage.dev` | https only |
| `RESEND_API_KEY` | Resend, sending access | `re_...` | Secret |
| `TRIAGE_EMAIL_FROM` | Sender, on a domain verified in Resend | `Triage <no-reply@gettriage.dev>` | No quotes around the value |
| `TRIAGE_EMAIL_LOGO_URL` | Optional logo in emails | `https://gettriage.dev/email/logo.png` | |
| `TRIAGE_REQUEST_LOG_RETENTION_DAYS` | Request-log retention | empty (365) | Keep the default; the privacy policy says 12 months |
| `TRIAGE_PUBLIC_STATS_MIN_REQUESTS` | Overview tab threshold | empty (100) | Keep the default in production |
| `TRIAGE_KEYLESS_LOOKUP_MINUTES` | How long a keyless request's cost can be read back | empty (10) | |

Set in the image or the compose file, not in `.env`: `TRIAGE_TRUSTED_PROXY_HOPS=1` (section 8), `HF_HOME`,
`HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `MALLOC_ARENA_MAX`, `OMP_NUM_THREADS`, `TRIAGE_TORCH_THREADS` and
`TRIAGE_SERVER_PROVIDERS=groq,gemini` (no Ollama on the server).

Never set in production: `TRIAGE_FAKE_MODELS`, `TRIAGE_FAKE_DB`, `TRIAGE_DEV_EMAIL_LOG`, `TRIAGE_TEST_DATABASE_URL`,
`TRIAGE_CACHE_PATH`, `TRIAGE_LOG`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `OPENROUTER_API_KEY`, `OLLAMA_BASE_URL`
(paid providers are used only with a user's own key).

Only `API_DOMAIN` and `ACME_EMAIL` reach the Caddy container; the API container reads the whole file.

### Generating the secrets

Run each on your own machine and paste the output into `.env` on the server. Generate them fresh for production;
never reuse a development value.

```bash
# TRIAGE_HASH_SECRET
python3 -c "import secrets; print(secrets.token_hex(32))"

# TRIAGE_ENCRYPTION_KEYS (prefix the output with "k1:")
python3 -c "import base64, secrets; print(base64.b64encode(secrets.token_bytes(32)).decode())"

# TRIAGE_API_KEYS and TRIAGE_ADMIN_KEYS (one per key)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Keep a copy of `TRIAGE_HASH_SECRET` and `TRIAGE_ENCRYPTION_KEYS` somewhere safe (a password manager): losing the
encryption keys makes every saved provider key unreadable.

### Vercel

| Name | Purpose | Example format | Notes |
|---|---|---|---|
| `VITE_TRIAGE_URL` | The API's address, without `/v1` | `https://api.gettriage.dev` | Read at **build** time: redeploy after changing it |

Project settings: **Root Directory** `web`; keep **"Include files outside the root directory in the Build Step"**
on (the build copies `CASE_STUDY.md`, `docs/` and `reports/` images from the repo root); Node.js 22 or newer.
`web/vercel.json` sets the build and sends every route to `index.html`, so `/chat/<id>`, `/settings/api-keys` and
`/case-study` load on refresh; real files (`/assets/...`, `/case-study-media/...`) are served first.

## 5. First deploy

On the VM:

```bash
git clone <repository URL> triage && cd triage
cp deploy/.env.example .env && chmod 600 .env && nano .env      # section 4
./deploy/update.sh
```

`deploy/update.sh` builds the image (several minutes the first time on 1 OCPU: it downloads and installs every
package and the embedding model), starts the API and Caddy, and waits until `/healthz` reports healthy.

Then point DNS at the VM: at your DNS provider, add an **A record** `api` (for `api.gettriage.dev`) with the VM's
reserved public IP. Once it resolves (`dig +short api.gettriage.dev`), Caddy gets its certificate by itself within a
minute; `docker compose logs caddy` shows "certificate obtained successfully".

What runs:

- **The API container** (`Dockerfile`): every package from `requirements.lock` at the exact tested version (CPU-only
  torch), the embedding model baked in at build time, and the network never used for models at runtime
  (`HF_HUB_OFFLINE=1`). It runs as a plain user on a read-only filesystem (only the response cache's volume is
  writable). On start it runs `alembic upgrade head` (on `DATABASE_URL_UNPOOLED`), then uvicorn with **one worker**:
  the demo, sign-in and per-minute limits, the circuit breakers and the response cache live in that process's
  memory. `--no-access-log` keeps client IPs out of the logs, as the privacy policy promises. It isn't reachable from
  outside: only Caddy can talk to it, over Docker's private network.
- **Caddy**: the only thing listening on 80 and 443. It gets and renews the certificate, redirects HTTP to HTTPS,
  streams answers through as they arrive, and writes no access log.
- **Health checks:** two endpoints, neither needing a key and both returning status words only.
  - `GET /healthz` (liveness): 200 once the classifier and embedding model are loaded, else 503. It **never touches
    the database**. Docker checks it every 30 seconds, and Caddy starts only once it passes. Point any uptime
    monitor here.
  - `GET /readyz` (readiness): the same, plus `SELECT 1` on the database within 2 seconds. A result is reused for
    10 seconds. Use it for the smoke test and manual checks only: **never point a monitor at `/readyz`**, since every
    check can wake Neon (section 3). The first check after Neon has slept can take a second longer than usual, and
    occasionally a 503; check again.
- **Restarts and logs:** both containers restart unless you stop them, including after a reboot. Each keeps at most
  3 log files of 10 MB.

## 6. Updating and rolling back

**Update:**

```bash
cd ~/triage && ./deploy/update.sh
```

It checks out `main`, runs `git pull`, builds an image tagged with the new commit, restarts, and waits for a healthy
API. Then it deletes old images to keep the disk from filling up: it keeps the image now running and the one before
it (for rollback), removes every other `triage-api` image, and prunes unused layers and build cache
(`docker image prune`, `docker builder prune`).

**Roll back** to the previous image, without rebuilding:

```bash
cd ~/triage && ./deploy/rollback.sh
```

It switches to the previous image and checks out that commit (so the compose file and Caddyfile match). The next
`./deploy/update.sh` goes back to `main`.

**Database changes don't roll back by themselves.** Each start runs `alembic upgrade head`, which only moves forward.
If the release you're undoing added a migration, first downgrade the database while the new image is still running:
`docker compose exec api alembic downgrade -1`, then roll back. Most releases add no migration
(`git diff <old>..<new> -- migrations/versions` is empty).

Useful commands: `docker compose ps`, `docker compose logs -f api`, `docker compose restart api`, `df -h /`.

## 7. Memory

Measured on Linux arm64, natively, with the production image and `docker-compose.yml` (one worker,
`MALLOC_ARENA_MAX=2`, one torch thread), after routed requests including 18,000-character prompts:

| What | Memory |
|---|---|
| API process, after requests (peak) | 649 MB (671 MB) |
| Caddy | 20 MB |
| Whole Docker host, increase when the stack starts | 741 MB |

Where the API's memory goes: its own imports (FastAPI, SQLAlchemy, provider SDKs) ~130 MB, torch ~200 MB,
transformers ~25 MB, the embedding model ~160 MB, embedding a long prompt ~60 MB. It doesn't shrink when idle.

On a 3 GB VM the stack alone is about 24% of memory, before Ubuntu, Oracle's cloud agent and Docker's own daemons,
which add a few hundred MB more. So the VM stays clearly above the 20% idle line (section 1) and has plenty of room
below 3 GB. The compose file caps the API at 1.5 GB so a runaway process can't starve the VM. Check the real numbers on
the VM with `free -m` and `docker stats --no-stream`.

The memory reductions already in place: CPU-only torch, one torch thread, `MALLOC_ARENA_MAX=2`, long prompts embedded
4 chunks at a time, the classifier run as numpy instead of scikit-learn, and the embedding model run on plain
`transformers` instead of sentence-transformers. Tests check both against the originals: classifier probabilities
within 1e-9, embeddings within 1e-6, and identical routing decisions at every quality preset. Running the embedding
model on ONNX Runtime would cut about 200 MB more; it's an optional future optimization.

## 8. Client IPs behind Caddy: `TRIAGE_TRUSTED_PROXY_HOPS=1`

The demo's per-visitor limit and sign-in rate limits use the visitor's IP (stored only as an HMAC). Behind a proxy
the API sees the proxy's address, and the visitor's arrives in `X-Forwarded-For`. `TRIAGE_TRUSTED_PROXY_HOPS=N` takes
the N-th address from the right, so anything a client adds itself (further left) can't choose the address.

Caddy is the only proxy in front of the API, and it sets `X-Forwarded-For` to the address it received the connection
from, ignoring any value the client sent (Caddy trusts no upstream proxies by default). So the rightmost entry is the
visitor, which is what `1` reads. `0` would make every visitor look like Caddy's container and share one allowance;
`2` would let a client pick its own address. `docker-compose.yml` sets the value, since it follows from that layout.
If you ever put another proxy or CDN in front of Caddy, this changes.

Check it after the first deploy:

1. **A made-up header isn't trusted:** send one demo chat message from your computer, then compare
   `curl -s https://api.gettriage.dev/v1/demo/status` with
   `curl -s -H "X-Forwarded-For: 203.0.113.7" https://api.gettriage.dev/v1/demo/status`. Both should show the same
   `remaining`.
2. **Visitors are told apart:** open the demo from a phone on mobile data. Its allowance should be full.

## 9. Resend (email)

1. Add and verify the sending domain (the DNS records Resend lists).
2. Create an API key with sending access only: `RESEND_API_KEY`.
3. Set `TRIAGE_EMAIL_FROM` to an address on that domain. A bad sender is caught at startup with a warning naming
   the variable.

## 10. Smoke test after each deploy

- [ ] `curl -s https://api.gettriage.dev/readyz` returns `"status": "ok"` with `"database": "ok"`;
      `docker compose ps` shows the API healthy.
- [ ] `docker compose logs api`: no "refusing to start", no warnings about the email sender or logo.
- [ ] `http://api.gettriage.dev` redirects to https, and the certificate is valid.
- [ ] Website loads; refresh `/case-study`, `/chat` and `/settings/api-keys` directly: each loads (no 404).
- [ ] Browser console on the website: no CORS errors.
- [ ] Demo chat (logged out): an answer streams in, the routing panel shows model, tier and cost.
- [ ] Press Stop during a long answer: it stops and later shows an estimated cost.
- [ ] A long answer with a table renders fully (no "stopped part-way").
- [ ] Sign up: the welcome email arrives, its links point at the website.
- [ ] Log out, "Forgot password": the code arrives and resets the password.
- [ ] Save a provider key in Settings: it's checked and shown by its last 4 characters; chat uses it.
- [ ] Create a Triage API key and call the API with the OpenAI SDK (`model="auto"`): 200, `X-Triage-*` headers.
- [ ] Dashboard: Benchmark shows the test results; My usage shows the requests just made.
- [ ] Client-IP check (section 8).
- [ ] Neon: the tables exist (`alembic_version` at the latest revision).
- [ ] Neon: the production compute is a fixed 0.25 CU, and after 10 minutes with no visitors the console
      shows it suspended (idle).
- [ ] On the VM: `free -m` and `df -h /` look healthy.

## 11. Maintenance

**Updating dependencies.** Change a pin in `requirements.txt`, reinstall locally, run the tests
(`set -o pipefail; venv/bin/python -m pytest` and `-m postgres`), then regenerate `requirements.lock` on Linux with
your tested versions as constraints:

```bash
venv/bin/pip freeze --exclude pip | grep -v -i -E "^(torch|setuptools)==" > /tmp/constraints.txt
docker run --rm --platform linux/arm64 -v "$PWD:/src" -v /tmp/constraints.txt:/c.txt python:3.14.7-slim sh -c '
  pip install -q torch==2.15.0+cpu --index-url https://download.pytorch.org/whl/cpu &&
  pip install -q -r /src/requirements.txt -c /c.txt && pip check &&
  pip freeze --exclude pip --exclude setuptools' | sort -f
```

Put that output under the header lines of `requirements.lock` (they keep the CPU index line). The lock installs
identically on arm64 and x86_64: PyTorch's CPU index has `torch 2.15.0+cpu` for both, and the arm64 image's installed
packages match the lock exactly (only `setuptools`, which comes with the Python image, is extra).

**Retraining.** A retrained classifier gets new `.joblib`/`.json`/`.npz` files. Commit all three for the version
`config/routing.yaml` names, un-ignore its `.joblib` in `.gitignore` the way v2's are, and add the files to
`.dockerignore`'s allow-list.

**npm audit.** `npm audit --omit=dev` in `web/` reports 6 moderate findings and nothing high or critical. They come
from two build-time tools, not code shipped to browsers: `minimatch` (through the shadcn CLI) and
`postcss-selector-parser` (through `@tailwindcss/typography`). No action needed; recheck when updating those packages.
