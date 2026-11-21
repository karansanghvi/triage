## Introduction

Triage is an OpenAI-compatible router that sends each AI request to the cheapest model that can answer it well. It is a personal project run by Karan Sanghvi, an individual developer in Seattle, Washington, USA ("Triage", "we", "us").

This policy explains what information Triage collects when you visit gettriage.dev, try the free demo, create an account, or call the Triage API, how that information is used, and the choices you have. By using Triage, you agree to the practices described here.

## Information we collect

Triage collects only what it needs to route your requests and run your account. Secrets are never stored in readable form.

| What | Details | How we store it |
| --- | --- | --- |
| Account details | Your email address, display name, preferred provider and when you joined | In our database |
| Password | Used only to log you in | Only as a one-way hash (argon2id); we can never read it |
| Login sessions | A session token, created when you log in | Only as a hash on our side; the token itself stays in your browser |
| Provider keys | Your OpenAI, Anthropic, Groq or Gemini API keys, if you save them | Encrypted (see "Your provider keys") |
| Triage API keys | The name you give each key, its short public ID, and when it was created, last used or revoked | The secret part only as a keyed hash; it is shown to you once |
| Request records | For each request: the time, the model and tier that answered, the routing scores, the quality setting, token counts, estimated cost, how long it took, whether it succeeded (with a short error message if not), a keyed fingerprint of the conversation (not its text), and, for signed-in requests, which account made it | In our database, for up to 12 months |
| Free demo visitors | A pseudonymous identifier derived from your IP address, used to apply the free demo limit (10 requests a day) and the rate limits on signing up, logging in and password resets | As a keyed hash, not your raw IP address |
| Password reset codes | A 6-digit code we email you | Only as a hash; expires after 10 minutes |

**Prompt and answer text.** Triage doesn't store the text of your prompts. To avoid paying twice for the same question, it keeps a copy of each answer for up to 24 hours. Free demo requests, requests made on your own keys, through your account or with a Triage API key are never used to train Triage's routing model.

**Stored only in your browser.** Your chat history, the name you give the demo and your login session are kept in your browser's local storage. Your chat history is never stored on our servers. Each message you send includes the conversation so far, so the model has context; we pass it to the AI provider and don't keep its text. You can clear chats in Settings → Account, or by clearing your browser's site data.

## How we use information

We use the information above only to run and improve Triage:

- **Answer your requests:** choose a model, send your messages to that provider, and return the answer with its routing details.
- **Run your account:** log you in, keep your keys and settings, and let you manage them.
- **Keep the service safe and fair:** apply rate limits and the free demo limit, prevent abuse, and investigate errors.
- **Send service emails:** a welcome email when you sign up, and password reset codes. We don't send marketing emails.
- **Show usage and results:** your own usage in your dashboard, and overall statistics about Triage's routing that never identify you.
- **Improve routing:** only requests made with Triage's own internal API keys are used to improve the routing model. Free demo requests, requests through your account and requests on your own keys never are.

## Your provider keys

Your provider keys are encrypted before they are saved and are never shown again in full.

- **Encrypted at rest:** each key is encrypted with AES-256-GCM before it reaches our database.
- **Used only for your requests:** a key is decrypted in memory only to answer a request you made, to check it when you ask us to, or when we rotate our encryption keys, and is never written to logs.
- **Only the last 4 characters are shown:** that is all you, or anyone viewing your account, can see after saving.
- **Checked when you save it:** we confirm a key works by calling that provider's free list-models endpoint. This does not use your quota for answers.
- **You stay in control:** remove a key at any time in Settings → Provider keys. Deleting your account deletes all of them.

Usage on your own keys is billed by the provider to your provider account, under that provider's terms.

## AI providers and service providers

To answer a request, Triage sends your messages to the AI provider it chooses. That provider handles them under its own privacy policy and terms.

| Provider | When your messages go there |
| --- | --- |
| OpenAI | Only when you have saved an OpenAI key, or sent one with your request |
| Anthropic | Only when you have saved an Anthropic key, or sent one with your request |
| Groq | On the free demo (our key), or when you have saved or sent a Groq key |
| Google (Gemini) | On the free demo (our key), or when you have saved or sent a Gemini key |

If you turn on answer checking for an API request, Triage also sends the prompt and its answer to a checking model (on Groq), using your own key for account requests, and may ask a stronger model to answer again. The free demo never uses answer checking.

We also rely on these service providers to run Triage. They process data only to provide their service to us:

| Service provider | What it does for Triage |
| --- | --- |
| Neon | Hosts our database (United States) |
| Oracle Cloud Infrastructure | Runs the Triage API |
| Vercel | Hosts the gettriage.dev website |
| Resend | Sends our welcome and password reset emails |

## What we don't do

- We don't sell or rent your personal information, or share it for advertising.
- We don't show ads.
- We don't use advertising or cross-site tracking cookies. Triage uses your browser's local storage only to make the app work.
- We don't use free demo requests, or requests made on your own keys, through your account or with a Triage API key, to train any model.
- We never look at your provider keys; they are decrypted only by the service, to make the calls you ask for. We can't see your password.

## Security

We protect your information with measures that fit a service handling API keys:

- All traffic to Triage uses HTTPS.
- Passwords, session tokens, Triage API keys and reset codes are stored only as hashes.
- Provider keys are encrypted with AES-256-GCM, and key-shaped text is removed from our logs.
- Only the person who runs Triage can access its systems.

No system is perfectly secure. If a breach affects your information, we will notify you as the law requires.

## Retention and deletion

We keep your account information until you delete your account.

- **Deleting your account** (Settings → Account) permanently removes your account details, provider keys, Triage API keys and login sessions straight away. Request records stay for usage statistics, with the link to your account removed. They never contained your prompts or answers.
- **Reset codes** expire after 10 minutes, and reset links after 15 minutes.
- **Free demo limits** reset every day.
- **Free demo request records** are kept for usage statistics. They contain no prompt or answer text and no IP address.
- **Cached answers** are deleted after 24 hours.
- **Request records** are deleted after 12 months.
- **Server logs** kept by our hosting provider may include your IP address for a limited time.
- **Backups** made by our database provider may hold deleted data for up to 6 hours before they expire.

Information in your browser stays until you clear it in Settings → Account or in your browser.

## Your choices and rights

You can manage most of your information yourself in Settings:

- Change your display name or password.
- Add, re-check, replace or remove provider keys.
- Create or revoke Triage API keys.
- Clear your chat history from this browser.
- Delete your account and everything linked to it.

Depending on where you live, for example in the European Union, the United Kingdom or California, you may also have the right to access, correct, delete or receive a copy of your personal information, and to object to some uses of it. To make a request, email us (see "Contact us"). We will respond within 30 days and won't treat you differently for using these rights.

## Children

Triage is a tool for developers and is not meant for children. We don't knowingly collect information from anyone under 13, or under 16 in the European Economic Area and the United Kingdom. If we learn that we have, we will delete it.

## Where your information is processed

Triage runs in the United States. If you use it from another country, your information is transferred to and processed in the United States.

## Changes to this policy

We will post any changes on this page and update the date at the top. If a change materially affects how we use your information, we will also email account holders before it takes effect.

## Contact us

For privacy questions or requests, email karansanghvi1303@gmail.com.
