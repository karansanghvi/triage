# Reference review decisions (Phase 2 Step 3)

Review of the 28 prompts flagged by `python -m eval.score_answers` on 2026-10-20: prompts where every
label tier (low, medium, high) failed, or where low and medium passed but high failed. Decided on
2026-10-20. Applied on 2026-10-22 (see "To apply" below).

## Rule

- Benchmark records (GSM8K, MMLU, Arena) are never edited. If one is wrong or ambiguous, it is excluded
  and a replacement is drawn from the same subject/split, the same way as `mmlu_0102` (row 3895).
- Own prompts are reworded to remove the ambiguity, keeping the reference.
- A record is only excluded with an independent reason the question is broken (the text, options or
  arithmetic), never just because models failed it.

"Decided by": **Karan** = decided by the project owner; **Karan (proposed by Claude)** = checked and
proposed by Claude under the rule above, then reviewed and confirmed by Karan. All 28 decisions were
reviewed and confirmed by Karan on 2026-10-22.

## Exclude and replace (benchmark records)

| Record | Source row | Reason | Decided by |
|---|---|---|---|
| `gsm8k_0001` | test row 1309 | Reference is wrong. The dataset's own solution adds Maryam's raise (400) instead of Sarah's (300): "750+430+400+700 = 2280". Correct total is 430 + 750 + 300 + 700 = **2,180**. Rechecked against the original question and solution text. | Karan (proposed by Claude) |
| `gsm8k_0045` | test row 255 | Question contradicts itself: "Ten stalls have 20 cows each" … "into each of the twenty stalls". | Karan (proposed by Claude) |
| `gsm8k_0089` | test row 1161 | Ambiguous: "another 1/4 of his land" can mean a quarter of the original 80 m² (answer 140) or of the remaining land (answer 170, the reference). | Karan (proposed by Claude) |
| `gsm8k_0173` | test row 1059 | Ambiguous: "feeds her cats twice a day with 60 grams" doesn't say per cat (2 days, the reference) or in total (6 days). | Karan (proposed by Claude) |
| `mmlu_0030` | virology, row 13733 | PrEP is effective in both women and drug users, so "D) a. and b. above" is correct; the reference "B) Drug users" is incomplete. | Karan (proposed by Claude) |
| `mmlu_0172` | college_chemistry, row 910 | Spin trapping is used because free-radical steady-state concentrations are too low for direct EPR detection (C); the reference D (lower power) is not the accepted reason. | Karan (proposed by Claude) |
| `mmlu_0153` | astronomy, row 263 | Two defensible options: the 1:2:4 resonance keeps Io's orbit elliptical (B, the reference), which is exactly what produces the tidal heating in A. | Karan |
| `mmlu_0166` | international_law, row 6457 | Two defensible options: standard-setting as proposing non-binding standards (C, the reference) versus setting standards of conduct in treaties (D). | Karan |
| `mmlu_0059` | high_school_us_history, row 5654 | Reference B ("Washington … reluctantly accepted the endorsement of the Federalist party") is historically false; Washington never accepted a party endorsement. A (parties formed from post-Constitution debates: Federalists and Democratic-Republicans) is the accurate statement. | Karan (proposed by Claude) |
| `mmlu_0068` | business_ethics, row 481 | Broken question: the stem is only "How the code is enforced." and the options ("1,2,3", "1,2,4", …) refer to numbered statements that are missing from the text. | Karan (proposed by Claude) |
| `mmlu_0171` | college_chemistry, row 977 | Reference looks wrong and an option is malformed. In the high-temperature limit the polarization ratio is \|γe\|/γH = 1.761×10¹¹ / 2.675×10⁸ ≈ **658** (option C), not 820 (the reference). Option A is not an answer at all; it is the given constant "[\|ɣe\| = 1.761 x 10^11 T^-1 s^-1]". | Karan (proposed by Claude) |
| `mmlu_0180` | electrical_engineering, row 1948 | Reference is wrong. For a step-up (boost) chopper, Vo = Vin / (1 − D), so D = 1 − 110/150 = **0.27** (option D), not 0.67 (the reference). | Karan (proposed by Claude) |

## Reword (own prompts, references unchanged)

| Record | Change | Decided by |
|---|---|---|
| `own_extraction_016` | State that `shipper` and `consignee` are the company names only, with no addresses. All tiers included the full addresses, which the current wording allows. | Karan (proposed by Claude) |
| `own_extraction_027` | State that `os_version` is the version number only (e.g. "14.6", not "Sonoma 14.6"). | Karan (proposed by Claude) |

## Keep as is

| Record | Reason | Decided by |
|---|---|---|
| `mmlu_0082` | Reference correct: A, B and C are each true (Option II has the highest expected return; Option I guarantees ≥$20K; only Option III can reach $80K), so "D) All of the above" is right. | Karan (proposed by Claude) |
| `mmlu_0010` | Reference correct: in Esman's linkage model, organisations sharing industrial interests (e.g. trade associations) are normative publics (C). | Karan (proposed by Claude) |
| `own_reasoning_025` | Reference verified: 2026-10-16 20:30 UTC is Friday 15:30 in Chicago; 1.5 business hours Friday + 6.5 Monday = 2026-10-19 15:30. A deliberate hard trap. | Karan (proposed by Claude) |
| `mmlu_0020` | Low and medium agree with the reference; no independent sign the question is broken. | Karan (proposed by Claude) |
| `mmlu_0076` | Low and medium agree with the reference; no independent sign the question is broken. | Karan (proposed by Claude) |
| `mmlu_0133` | Low and medium agree with the reference; no independent sign the question is broken. | Karan (proposed by Claude) |
| `mmlu_0142` | Low and medium agree with the reference; no independent sign the question is broken. | Karan (proposed by Claude) |
| `mmlu_0004` | Reference correct: "intelligent → succeed; these people aren't intelligent → they won't succeed" is denying the antecedent (C). | Karan (proposed by Claude) |
| `mmlu_0016` | Reference correct: in OKWS, services inherit their oklogd socket from okld on fork/exec and open database-proxy connections over TCP themselves; fd passing is only needed to hand client connections to services. So both statements are false (B). | Karan (proposed by Claude) |
| `mmlu_0079` | Reference defensible: morphine oral solution 10 mg/5 ml (0.2%) is a Schedule 5 preparation under UK rules, exempt from controlled-drug requirements, unlike the codeine injection, sustained-release morphine and fentanyl patch (D). Country-specific (UK) but consistent. | Karan (proposed by Claude) |
| `mmlu_0095` | Reference correct: Vietnam (North/South), the UK (Hadrian's Wall) and Germany (East/West) all have well-known relict boundaries; Bolivia does not (D). | Karan (proposed by Claude) |
| `mmlu_0111` | Reference correct: Dershowitz uses the 9/11 shoot-down case to argue such decisions should be made openly, with accountability at the highest level (B), as in his torture-warrant proposal. | Karan (proposed by Claude) |
| `mmlu_0168` | Reference correct: removing a red sock from a roommate's whites is helpful, and ignoring a neighbour's test is not cheating, so both are not wrong (D). | Karan (proposed by Claude) |
| `mmlu_0176` | Reference defensible: the sociology textbook answer attributes 1980s inequality growth to rising unemployment and welfare dependence (D); the other options are false or irrelevant. | Karan (proposed by Claude) |

## To apply (after the collector finishes)

Applied on 2026-10-22, after collection finished.

- [x] Added the 8 MMLU rows above to `MMLU_EXCLUDED_ROWS` in `eval/benchmarks.py` (with a comment each) and
  rebuilt MMLU; replacements were drawn from the same subject in the same slot.
- [x] Added `GSM8K_EXCLUDED_ROWS` to `eval/benchmarks.py` with the 4 test rows above, drawing replacements from
  the test split in the same slot, and rebuilt GSM8K.
- [x] Reworded `own_extraction_016` (shipper and consignee are company names only, no addresses) and
  `own_extraction_027` (os_version is the version number only, without the release name) in
  `data/own_prompts.jsonl`; references unchanged.
- [x] Re-merged `data/prompts.jsonl` and ran the validator (0 errors). Exactly these 14 records changed; every
  other record, every ID and the train/test split (and its hash) are unchanged. Moved the 62 old answers for the
  14 IDs to `data/answers/stale_answers.jsonl` (kept, not deleted), collected new answers for the 14 prompts on
  every tier (`python -m eval.build_dataset --only-ids ...`), and re-scored.

Replacement rows drawn: `gsm8k_0001` 1309→1137, `gsm8k_0045` 255→594, `gsm8k_0089` 1161→1139,
`gsm8k_0173` 1059→978, `mmlu_0030` 13733→13769, `mmlu_0059` 5654→5720, `mmlu_0068` 481→424,
`mmlu_0153` 263→299, `mmlu_0166` 6457→6540, `mmlu_0171` 977→993, `mmlu_0172` 910→933, `mmlu_0180` 1948→1984.

## Follow-up: second `mmlu_0030` replacement

| Record | Source row | Reason | Decided by |
|---|---|---|---|
| `mmlu_0030` | virology, row 13769 (the first replacement) | The question asks what "zoonotic" means, but the reference "A) Such viruses are confined to animals" contradicts the definition: a zoonotic virus crosses from animals to humans. Excluded and replaced again from virology in the same slot. | Karan (flagged by Claude) |

Summary: 12 excluded and replaced (4 GSM8K, 8 MMLU), 2 reworded, 14 kept; plus 1 follow-up exclusion
(the first replacement for `mmlu_0030`).
