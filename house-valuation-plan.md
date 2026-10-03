# House Condition Rating & Price Estimation: Plan

Date: 2026-10-02

## Goals

1. Rate the interior condition of houses from images, reliably.
2. Differentiate houses by condition, luxury, etc.
3. Improve price estimation accuracy (currently a linear function with hand-set weights).

---

## 1. Rating interiors from images

### Approach
Use a vision-language model (VLM, e.g. Claude) with a structured rubric. Only train a custom model if this proves insufficient.

1. **Define the rubric first.** Most of the quality comes from this step.
2. **Prompt a VLM per photo**, requesting structured JSON: room type, per-axis scores, evidence, confidence. Ask for evidence before the score.
3. **Aggregate per listing.** Weight kitchens and baths more heavily; use min or low percentile for damage flags so one nice photo doesn't mask a bad room.
4. **Validate** on a hand-labeled set (100-300 listings). Measure weighted kappa or Spearman correlation. Iterate on rubric and prompt.
5. **Custom model (optional):** fine-tune a CLIP/ViT embedding with an ordinal/regression head only if there are thousands of reliable labels. VLM scores can bootstrap labels.

### Is Fannie Mae C1-C6 a proper rubric? Not on its own
- C1-C6 is Fannie Mae's UAD condition scale (not Zillow's). Definitions are public (UAD Appendix; Selling Guide B4-1.3-06). Actual per-house ratings are private to appraisers/lenders.
- Problems for photo-based scoring:
  - Partly unobservable from photos (structure, roof, mechanicals).
  - Conflates condition with modernity/renovation.
  - No luxury dimension, so it can't separate a well-kept starter home from a well-kept luxury home.
  - Vague anchors ("limited wear", "some deferred maintenance") give inconsistent scoring.
- Use it as a reference for the **condition axis only**, and as a final mapping (condition base, capped by visible damage).

### Recommended: separate photo-observable axes

| Axis | Measures | Observable cues |
|---|---|---|
| Condition | Physical state and wear | Stains, cracks, peeling paint, worn flooring, damaged fixtures, mold |
| Modernity | How recently updated | Cabinet/counter style, fixture finishes, lighting, flooring type |
| Finish quality | Material grade | Stone vs. laminate, custom vs. stock cabinetry, hardwood vs. LVP, trim detail |
| Luxury/design | Overall upscale-ness | Ceiling height, scale, built-ins, pro-grade appliances, architectural features |
| Layout/space (optional) | Openness and light | Natural light, open plan, room size cues |

Axes vary independently: dated-but-clean (high condition, low modernity), flip (high modernity, low finish quality), luxury with deferred maintenance (high luxury, low condition).

### Making it reliable
- Concrete observable anchors per axis and level (1-5); ideally paired with reference photos as few-shot examples.
- Score per room type, then aggregate. Kitchens and baths carry most signal.
- Checklist before score: list observed features, then assign levels.
- Allow "can't tell" (null + low confidence) when photos don't show enough.
- Test reliability: score the same house twice; have two humans label a sample. Low human agreement on an axis means the rubric is too vague there.

### Pitfalls
- Listing photo bias: curated, wide-angle, staged, edited, so scores inflate. Calibrate against labels.
- Missing rooms: track coverage; lower confidence when key rooms are absent.
- Score drift: low temperature, pin model version, keep a fixed anchor set.
- Fair housing: restrict to physical property condition; no comments on occupants or neighborhood.
- Clutter/cleanliness should be scored separately from condition.

### Validating that it differentiates houses
Correlate scores with independent signals: price per sqft within a market, year built/renovated, permit history. Luxury and finish quality should track price per sqft; condition should track age and deferred-maintenance signals.

---

## 2. Improving price estimation

Core fix: **learn weights from sold-price data instead of hand-setting them.**

### Steps
1. **Ground truth:** use sold prices (not list prices, which are biased high), with sale date, location, and features (including image scores) as of sale time.
2. **Baseline:** regularized (ridge/lasso) regression on `log(price)`. Log because effects are multiplicative.
3. **Boosted trees** (LightGBM/XGBoost) to capture interactions (e.g. luxury matters more in bigger houses, condition effects vary by neighborhood). Keep the linear model as a baseline; only switch if boosting clearly wins on held-out data.
4. **Location and size dominate.** Get these right before tuning image features:
   - Sqft, lot size, beds/baths, year built
   - Neighborhood/ZIP target encoding or lat/lon spatial features
   - Comparable sales: median price per sqft of nearby recent sales (often the strongest single feature)
   - Sale date or market index
   Image scores then act as a residual adjustment (why a house sold above or below its location/size prediction).
5. **Evaluate honestly:**
   - Split by time, not randomly (random splits leak market trends and overstate accuracy).
   - Report median absolute percentage error (MdAPE) and share within 5%/10%/20% of sale price.
   - Ablation: with vs. without image features. If MdAPE doesn't improve, the image scores aren't adding value.
   - Check errors by segment (price tier, neighborhood, property type); luxury and unusual homes are usually worst.
6. **Uncertainty:** output ranges (quantile regression or conformal prediction), not just point estimates. Wide ranges flag low-trust properties (few photos, no comps).

---

## 3. Our data: sold prices for the past 2 and 5 years

- **5-year window:** use for fitting (more data for location/size effects and rare segments).
- **2-year window:** the "recent market" reference.
- **Time handling:** 5 years spans real market shifts. Preferred: time-adjust historical prices to today's dollars with a local index (e.g. monthly median price per sqft by ZIP); optionally add recency sample weights.
- **Split:** train on oldest, validate on middle, test on the most recent ~6 months. Also compare a 5-year time-adjusted model against a 2-year-only model.

### Open questions / data needs
- Volume: sales per market and number of markets. About 1,000+ per market allows per-market models; a few hundred means pooling with location features.
- Photos at time of sale: image scores are only valid if photos match the property's condition at sale (post-renovation or later photos corrupt the signal).
- Fields available: sqft, beds/baths, lot size, year built, coordinates/address, sale date, list price.
- Data quality: flag or exclude non-arm's-length sales (family transfers, foreclosures, auctions).
- Where does the data live, and in what format?

---

## Suggested order of work

1. Draft the multi-axis rubric with level anchors (and few-shot reference photos).
2. Run a first VLM pass on a sample; measure repeatability and human agreement.
3. Collect sold-price data with features; time-adjust prices.
4. Fit location+size-only log-linear baseline; record MdAPE on the time-based test set.
5. Add image scores; measure the improvement (ablation).
6. Try gradient boosting; compare to the linear model.
7. Add uncertainty intervals; analyze errors by segment.
