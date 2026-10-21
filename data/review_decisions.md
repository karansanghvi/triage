# Reference review decisions (Phase 2 Step 3)

Review of the 28 prompts flagged by `python -m eval.score_answers` on 2026-10-20: prompts where every
label tier (low, medium, high) failed, or where low and medium passed but high failed. Decided on
2026-10-20. **Not applied yet**: the collector is still running; changes are applied after it finishes.

## Rule

- Benchmark records (GSM8K, MMLU, Arena) are never edited. If one is wrong or ambiguous, it is excluded
  and a replacement is drawn from the same subject/split, the same way as `mmlu_0102` (row 3895).
- Own prompts are reworded to remove the ambiguity, keeping the reference.
- A record is only excluded with an independent reason the question is broken (the text, options or
  arithmetic), never just because models failed it.

"Decided by": **Karan** = decided by the project owner; **Claude** = checked and proposed by Claude
under the rule above (pending Karan's review). Where Karan confirmed a proposal from the first review
pass, both are listed.

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
| `mmlu_0059` | high_school_us_history, row 5654 | Reference B ("Washington … reluctantly accepted the endorsement of the Federalist party") is historically false; Washington never accepted a party endorsement. A (parties formed from post-Constitution debates: Federalists and Democratic-Republicans) is the accurate statement. | Claude |
| `mmlu_0068` | business_ethics, row 481 | Broken question: the stem is only "How the code is enforced." and the options ("1,2,3", "1,2,4", …) refer to numbered statements that are missing from the text. | Claude |
| `mmlu_0171` | college_chemistry, row 977 | Reference looks wrong and an option is malformed. In the high-temperature limit the polarization ratio is \|γe\|/γH = 1.761×10¹¹ / 2.675×10⁸ ≈ **658** (option C), not 820 (the reference). Option A is not an answer at all; it is the given constant "[\|ɣe\| = 1.761 x 10^11 T^-1 s^-1]". | Claude |
| `mmlu_0180` | electrical_engineering, row 1948 | Reference is wrong. For a step-up (boost) chopper, Vo = Vin / (1 − D), so D = 1 − 110/150 = **0.27** (option D), not 0.67 (the reference). | Claude |

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
| `mmlu_0004` | Reference correct: "intelligent → succeed; these people aren't intelligent → they won't succeed" is denying the antecedent (C). | Claude |
| `mmlu_0016` | Reference correct: in OKWS, services inherit their oklogd socket from okld on fork/exec and open database-proxy connections over TCP themselves; fd passing is only needed to hand client connections to services. So both statements are false (B). | Claude |
| `mmlu_0079` | Reference defensible: morphine oral solution 10 mg/5 ml (0.2%) is a Schedule 5 preparation under UK rules, exempt from controlled-drug requirements, unlike the codeine injection, sustained-release morphine and fentanyl patch (D). Country-specific (UK) but consistent. | Claude |
| `mmlu_0095` | Reference correct: Vietnam (North/South), the UK (Hadrian's Wall) and Germany (East/West) all have well-known relict boundaries; Bolivia does not (D). | Claude |
| `mmlu_0111` | Reference correct: Dershowitz uses the 9/11 shoot-down case to argue such decisions should be made openly, with accountability at the highest level (B), as in his torture-warrant proposal. | Claude |
| `mmlu_0168` | Reference correct: removing a red sock from a roommate's whites is helpful, and ignoring a neighbour's test is not cheating, so both are not wrong (D). | Claude |
| `mmlu_0176` | Reference defensible: the sociology textbook answer attributes 1980s inequality growth to rising unemployment and welfare dependence (D); the other options are false or irrelevant. | Claude |

## To apply (after the collector finishes)

1. Add the 8 MMLU rows above to `MMLU_EXCLUDED_ROWS` in `eval/benchmarks.py` (with a comment each) and
   rebuild MMLU; replacements are drawn from the same subject in the same slot.
2. Add a GSM8K exclusion list to `eval/benchmarks.py` (the loader has none yet) with the 4 test rows
   above, drawing replacements from the test split in the same slot, and rebuild GSM8K.
3. Reword `own_extraction_016` and `own_extraction_027` in `data/own_prompts.jsonl`.
4. Re-merge `data/prompts.jsonl`, run the validator, collect answers for the 14 changed prompts on every
   tier, and re-score.

Summary: 12 excluded and replaced (4 GSM8K, 8 MMLU), 2 reworded, 14 kept.
