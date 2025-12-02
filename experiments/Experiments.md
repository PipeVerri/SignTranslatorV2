[English](Experiments.md), [Español](Experiments.es.md)

# How feasible is real-time translation from LSA to Spanish?

In this document I outline my reasoning process, what I tried, what failed, and how I plan to continue.

---

# Prior Research and State of the Art

The first major approach (the one I initially attempted to replicate) was:

1. **Train an RNN to identify individual glosses** from pre-segmented video.
2. Perform **heuristic real-time segmentation**: monitor acceleration, fixed-size windows, etc. to decide when a sign “occurred”.
3. Every time the heuristics detect “a sign happened here”, crop that segment and feed it to the RNN.

The problem with this approach is that heuristic segmentation has been **abandoned because it performs extremely poorly**. Main issues:

1. **Coarticulation**:  
   If I produce the sign for *“yo”* (point to myself) followed by *“voy”* (point forward):
   - When recorded individually, the arm starts from a resting position for each sign.
   - When produced consecutively, I point to myself and then point forward **without lowering or pausing the arm**.
2. The speed of a sign changes significantly when produced inside a phrase vs in isolation, and even between phrases depending on the signer’s tempo.

Then came **gloss-free models**:

- They perform *everything* inside a single transformer:
  - gloss segmentation,
  - interpreting gloss-level meaning,
  - gloss → text translation.
- They work well with large datasets, but with small datasets (like LSA-T) they perform worse because they must learn **three separate tasks simultaneously** with limited supervision.

The newest approaches are **semi-gloss-free**, closer to what I want to build with an LLM (explained later):

- They depend less on perfect gloss annotations.
- They leverage large language models for the textual component.
- They let the vision/sequence model focus more on “what is happening in the video”.

---

# Experiment 1: Simple RNN

## Dataset and Setup

I wanted something manageable to start with, so I used **LSA64**:
- 64 distinct signs.
- ~50 samples per sign.

The plan:

- Train an **RNN** that takes per-frame *landmarks* (pose/hands) as input,
- Output is a **softmax over the 64 possible signs**.
- For continuous signing:
  - Feed the model frames in real time.
  - Consider a sign “detected” only when some class crosses a probability **threshold**.

Additionally, I tested a more explicit real-time setup:
- Process videos at **6 fps**.
- Use **2-second windows** for real-time processing.
- Apply a **softmax threshold** to differentiate “performing sign X” vs “doing nothing”.

#### Why use a bare RNN instead of a Gated RNN (LSTM/GRU) if those were the pre-transformer SOTA?
I wanted to benchmark the simplest possible model. Gated RNNs were designed to address long-term memory instability in vanilla RNNs. Why was long-term memory not an issue here?

The [classic paper](https://ieeexplore.ieee.org/document/279181) shows that the probability of learning useful long-term dependencies drops sharply after ~17 time steps.  
Since LSA64 clips last ~2s recorded at 6 fps, this gives ~12 steps—below the threshold where plain RNNs typically fail.

## Architecture and Hyperparameters

- RNN with **5 hidden layers**.
- Hidden size **144** (landmark generator outputs 144 parameters per frame).
- **100 epochs** of training.
- Learning rate \(10^{-4}\).
- L2 regularization with \(\lambda = 10^{-3}\).

## Trial 1

Training on pre-segmented single-sign videos:

- Train accuracy > **99%**.
- Validation accuracy ~ **97%**.

For such a simple network, a small dataset, and no data augmentation, these results were excellent.  
Problems appeared when moving to continuous sequences:

- Performance dropped drastically.
- The model **never saw long sequences with multiple consecutive signs**, or idle periods.
- As a result, it **never learned to forget past information**.

To mitigate this, I tried:

- Feeding the sequence in **2-second sliding windows** with **0.1s stride**.

## Trial 2

This improved things slightly, but detecting the correct sign in continuous mode remained difficult. Main issues:

1. **Broken segmentation**
   - If the segmentation cropped a sign halfway, the model failed.
   - Everything depended on hitting the “correct” window.

2. **Softmax answers “which sign is it?”, not “is there a sign?”**
   - Even when idle, the model was **very certain** some sign was occurring.
   - For visually similar signs (*rojo* vs *amarillo*), a 60% “rojo” score doesn’t mean “no sign”—it means “not fully confident”.

At this point I shifted focus from segmentation to the classification problem “sign vs no-sign”.

### Initial idea for “no-sign”

I attempted:

- Include ~2-second clips from **TED talks** where the speaker moves their hands but is **not signing**.
- Add a **65th class** for “no-sign”.
- Train the RNN to:
  - identify the 64 signs,
  - map TED-type movements to “no-sign”.

## What I learned from Experiment 1

Key next steps:

- **Lower fps** to ~6  
  Reduces temporal depth; each sign (~2s) yields ~10 frames, which is manageable.

- **Use a gated RNN (LSTM/GRU)**  
  If sequences exceed ~10 frames, LSTMs or GRUs are better suited.

- **Mix LSA64 with “no-sign” clips**  
  Real-world “non-sign” motion should be explicitly modeled.

### Generalization concerns

LSA64 is too uniform:
- Same seating, same background, same conditions.
- This likely harms generalization.

Ideas to address this:

- A dedicated network that predicts whether the sign uses:
  - **one hand**, or
  - **two hands**.  
  Could also consider mirroring for left/right dominance.

For sign classification:
- A single classifier for both one- and two-hand signs, but:
  - for one-hand signs, zero out the non-dominant hand + pass a “missing” flag,  
  **or**
- Two separate classifiers:
  - one-hand signs,
  - two-hand signs.

I’m still unsure where to place the **“no-sign” logit**:
- early stage (detecting sign-type),  
- or later stage (fine-grained classifier).

---

# Experiment 2: More Expressive RNN + More Data

Goal: keep the same pipeline but:

- increase model capacity,  
- address “sign vs no-sign” with more diverse data.

## Data Processing

- Process at **6 fps**.
- TED-talk clips of **2 seconds** for the “no-sign” class.
- Normalize hand scale using **hand keypoints**.

## Model

- Same pipeline, but:
  - wider RNN: **300 hidden units**.  
    Increasing depth caused overfitting with no validation gains; width helped.

## Results

- Only TED-style presenter hand motions were classified as “no-sign”.
  - The model is clearly **biased** toward that distribution.
- Repeating a sign several times works reasonably well.
  - But the major issue is the **intermediate windows**:
    - They look nothing like “hands down”.
    - They are not clean signs.
    - The model classifies them as actual signs.

Conclusion:

> Continuous signing requires **temporal segmentation learning**.  
> Fixed windows are inadequate: sign duration varies, and transitions are crucial.

## Idea: Use LSA-T + Weak Segmentation

A more promising idea:

- Use continuous datasets like **LSA-T**, which are TV-news videos in LSA with Spanish subtitles (I don’t have direct gloss annotations).
- Convert continuous Spanish text into **gloss sequences** using an NLP model.
- Use these glosses for **weakly supervised temporal segmentation**:
  - no perfect alignments, but an approximate structure.

Then train something similar to the original RNN but with:
- more realistic temporal structure,
- no reliance on fixed, arbitrary windows.

A [similar paper](https://arxiv.org/abs/2505.15438) explores this direction.

## Things to Try

- Test the [**PCA-based augmentation**](data_augmentation/02_PCA_augmentation/README.md) or [kinematic motion augmentation](data_augmentation/01_kinematic_augmentation/README.md).
  - Might be unnecessary now—classification is not the main bottleneck.
- A **two-stage RNN** to reduce overfitting to irrelevant hand positions:
  - Stage 1: detect “sign-type (dominant-hand / two-hands) / no-sign”.
  - Stage 2: fine-grained classification.
- Explore **semi-supervised learning**:
  - manually annotate portions of LSA-T,
  - use them to improve alignment and segmentation.

---

# Experiment 3: Moving to LSA-T with Larger Models

I know I will need a gloss-free or semi–gloss-free model. I want to experiment by modifying the architecture used in [LSA-T](https://sedici.unlp.edu.ar/bitstream/handle/10915/176192/Documento_completo.pdf-PDFA.pdf?sequence=1&isAllowed=y).

In the original paper, the transformer is fed by the output of 3 temporal convolutions with kernel size 1.  
This means each frame’s entire pose is compressed into **only 3 parameters**, which seems like an excessive dimensionality reduction.

## What I Want to Try

- Use **~50 temporal features** instead of just 3.
- Or skip convolutions entirely:
  - feed raw landmarks directly into a sequential model (RNN/Transformer).

## Face and Expressions

Another concern:

- Feeding **raw facial landmarks** into the model.
- I prefer:
  - training or reusing a transformer/RNN that **interprets facial expression** (possibly using a pretrained sentiment model),
  - passing this **higher-level representation** to the main translation model.

In other words:
- one model handles “what the face is expressing”, and
- another focuses on “what sign is being performed and what it means”, without dealing with raw facial geometry.

---
