# Computer Vision & the Photographic Gaze
### A Workshop for Photography Students

---

## Section 1: Introduction to Computer Vision — How Machines Learn to See

### Slides

1. **What is Computer Vision?**
   - Definition: teaching machines to interpret and understand visual information from the world
   - Brief history: from edge detection (1960s) to deep learning (2010s–now)
   - Key distinction: human vision vs. machine vision — we see meaning, machines see math (pixels, matrices, probability)

2. **How Does It Actually Work?**
   - Images as numbers: pixels, RGB channels, matrices
   - Feature extraction: edges, textures, shapes — what a machine "notices"
   - From handcrafted features (SIFT, Haar cascades) to learned features (CNNs, transformers)
   - Training data: machines learn from what they are *shown* — this is where bias enters

3. **The Building Blocks**
   - Classification ("this is a cat"), Detection ("there is a cat here"), Segmentation ("these exact pixels are the cat")
   - Confidence scores: machines don't "know" — they guess with percentages
   - The role of labelled datasets (ImageNet, COCO, LAION) — who labels, what gets labelled, what gets left out

### Interactive Demo (Presenter-led)
- **Feature visualization:** Run a photo through a CNN and display what each layer "sees" (edges → textures → parts → objects) using GradCAM or a similar tool. Take a student's photograph and show it decomposed — first into raw pixel values, then edges, then textures, then the abstract "concepts" the model recognizes. This makes the invisible process of machine vision visible and tangible.

---

## Section 2: Computer Vision, Surveillance & Imagination
### (Drawing on Ruha Benjamin's *Race After Technology*)

### Slides

4. **The New Jim Code**
   - Ruha Benjamin's concept: technology that reinforces racial hierarchies under the guise of neutrality and progress
   - Computer vision is not a neutral observer — it inherits the biases of its creators and training data
   - The "imagined objectivity" of algorithmic systems

5. **Surveillance as Infrastructure**
   - Facial recognition in policing, border control, public spaces
   - Case studies: predictive policing, China's social credit system, Clearview AI
   - The asymmetry of surveillance: who watches, who is watched, who is misidentified
   - Disproportionate impact on Black, Brown, and marginalized communities

6. **Imagination & Default Settings**
   - Benjamin's idea of the "default" — who is imagined as the standard human when systems are designed?
   - Training data as a reflection of whose faces, bodies, and lives are considered "normal"
   - The imagination gap: what futures does this technology foreclose?
   - Connection to photography's own history of racial bias (Shirley cards, film stock calibrated to white skin)

7. **Resistance & Counter-Surveillance**
   - Artists and activists pushing back: Zach Blas (*Facial Weaponization Suite*), Adam Harvey (*CV Dazzle*, *HyperFace*), Joy Buolamwini (*Gender Shades*)
   - Legislative efforts: bans on facial recognition, GDPR, right to not be seen
   - The question for photographers: what does it mean to make images in an age of ubiquitous machine reading?

### Interactive Demo (Presenter-led)
- **Training data archaeology:** Use the "Excavating AI" methodology — pull up a subset of ImageNet or LAION and show what labels have been assigned to images of people. Let students browse the categories and see how human bodies are classified, often with offensive, reductive, or absurd labels. This directly demonstrates Benjamin's argument that "neutral" systems encode specific imaginations about who people are. Connect it to photography: these are *photographs* someone took — now fed into a machine and stripped of their original context.

---

## Section 3: When Machine Vision Fails

### Slides

8. **Failure Modes: A Taxonomy**
   - Misclassification: gorilla label incident (Google Photos, 2015)
   - Invisibility: systems that literally cannot see certain people (soap dispensers, pulse oximeters, self-driving cars)
   - Hallucination: seeing things that aren't there (adversarial examples, pareidolia for machines)
   - Overconfidence: high confidence, wrong answer

9. **Who Bears the Cost of Failure?**
   - Failures are not evenly distributed — they cluster along lines of race, gender, disability, age
   - Real-world consequences: wrongful arrests (Robert Williams, Porcha Woodruff), denied services, algorithmic discrimination
   - The "it's just a bug" defense vs. structural critique

10. **The Limits of Pattern Matching**
    - Context blindness: machines see pixels, not meaning, history, or emotion
    - The difference between recognition and understanding
    - Edge cases are not edge cases when they affect entire communities
    - What does it mean that a machine cannot understand a photograph the way a photographer intends it?

### Interactive Demo (Presenter-led)
- **"Stump the model" live challenge:** Run a classifier (e.g., MobileNet via TensorFlow.js in the browser) and invite students to hold up objects, strike poses, or frame shots with their phones designed to confuse it. Show the top-5 predictions and confidence scores on screen. Unusual angles, partial occlusion, ambiguous compositions, reflections — photography students will intuitively know how to break framing. Discuss: why did it fail? What was it "expecting"? This turns failure analysis into a creative exercise that leverages their existing skills.

---

## Section 4: Computer Vision, Self-Surveillance & Beauty

### Slides

11. **The Mirror that Watches Back**
    - The smartphone camera as a surveillance device turned inward
    - From the mirror to the selfie to the filter: a brief history of self-regard mediated by technology
    - The front-facing camera as the most intimate surveillance tool in history

12. **Beauty Filters & the Algorithmic Face**
    - How beauty filters work: facial landmark detection, mesh mapping, real-time transformation
    - The "default beautiful" face: what do filters converge toward? (Smaller nose, larger eyes, lighter skin, symmetry)
    - FaceTune, Instagram filters, TikTok beauty mode — the normalization of self-editing
    - Bold Glamour (TikTok, 2023) as a turning point: uncanny realism, impossible to distinguish from "real"

13. **Beauty Trends as Data Feedback Loops**
    - How engagement metrics shape what beauty "means" — algorithmic amplification of certain faces and bodies
    - The homogenization of beauty: when everyone filters toward the same face
    - "Instagram Face" as a statistical average, not a cultural choice
    - The global export of beauty standards through technology (skin lightening filters, double eyelid filters)
    - Connection to cosmetic surgery trends following filter aesthetics

14. **Self-Surveillance & the Disciplined Body**
    - Foucault's panopticon turned inward: we become our own watchers
    - The labor of self-presentation in the age of computer vision
    - Dysmorphia by design: when the filtered self becomes the "real" self
    - Body scanning, posture tracking, "glow-up" culture — CV as a tool of self-discipline
    - The photographer's dilemma: documenting vs. constructing the self

### Interactive Demo (Presenter-led)
- **Live filter deconstruction:** Using a face-mesh library (e.g., MediaPipe Face Mesh running in the browser), show the raw facial landmark points mapped onto a student volunteer's face in real time. Then layer on a beauty filter step-by-step: first the mesh, then the smoothing, then the reshaping, then the final "beautified" output. Pause at each stage. Students see that a "filter" is really a series of deliberate geometric and tonal decisions — smaller nose, bigger eyes, smoother skin, lighter tone. Ask: who decided these were "improvements"? This demystifies the magic and makes the ideology of the filter visible.

---

## Suggested Additional Sections

### Section 5: Computer Vision & the Politics of Datasets (Optional)

- Where does training data come from? (Scraped without consent, Mechanical Turk labelers, LAION controversy)
- The labor behind the machine: underpaid annotators, content moderators, the Global South as AI's hidden workforce (reference Kate Crawford's *Atlas of AI*)
- Photographers' images in training data without consent — direct relevance to photography students
- Opt-out movements: Have I Been Trained?, Spawning, Nightshade/Glaze

### Section 6: Creative Responses — Artists Using (and Misusing) Computer Vision (Optional)

- Trevor Paglen: training sets as portraiture, machine-readable landscapes
- Hito Steyerl: proxy politics, how images operate beyond human perception
- Mimi Onuoha: *The Library of Missing Datasets* — what isn't seen is as political as what is
- Refik Anadol: large-scale data sculptures using machine vision
- Student prompt: How would you use (or refuse) computer vision in your photographic practice?

### Section 7: Looking Forward — Multimodal AI and the Future of the Image (Optional)

- CLIP, GPT-4V, Gemini: models that see *and* speak — what happens when machines describe photographs?
- AI-generated images (DALL-E, Midjourney, Stable Diffusion) — when machines make images, not just read them
- Provenance, authenticity, and trust: C2PA, watermarking, the crisis of photographic truth
- What role does the photographer play when machines can both see and create?

---

## Workshop Flow Summary

| Section | Theme | Duration (suggested) |
|---|---|---|
| 1 | Introduction to Computer Vision | 15–20 min |
| 2 | Surveillance & Imagination (Ruha Benjamin) | 20–25 min |
| 3 | When Machine Vision Fails | 15–20 min |
| 4 | Self-Surveillance & Beauty | 20–25 min |
| 5* | Politics of Datasets | 10–15 min |
| 6* | Creative Responses / Artists | 10–15 min |
| 7* | Multimodal AI & the Future of the Image | 10–15 min |

*Optional sections — include based on time and audience interest.

---

## Universal Interactive Demo: "How Does the Machine See Your Photo?"

> **A single web app that every student can open on their phone — no coding, no installation, no accounts.**

### Concept

A lightweight web app hosted on **GitHub Pages** (static, free, no backend needed) that lets students upload or take a photo and see how a machine "reads" it — combining elements from all four sections into one playful, provocative experience.

### What It Does (Step-by-Step Flow)

1. **Upload / Take a Photo** — Students use their phone camera or gallery. The image never leaves their device (all processing runs client-side via TensorFlow.js and MediaPipe).

2. **Step 1: "What the machine sees"** (Section 1)
   - The image is broken down into its pixel grid, showing RGB values on hover/tap
   - A heatmap overlay (GradCAM-style) shows which parts of the image the model is "paying attention to"
   - The model's top-5 classification labels are displayed with confidence percentages

3. **Step 2: "Who is watching?"** (Section 2)
   - If a face is detected, it draws a bounding box and displays estimated attributes (age, gender, emotion) — the same kind of metadata surveillance systems extract
   - A short text overlay appears: *"This is what a surveillance camera sees. No context. No consent. Just data points."*
   - Optional: show how many times per day the average person is captured by CCTV (with a counter animation)

4. **Step 3: "Where it breaks"** (Section 3)
   - Students are prompted to retake the photo in a way that might confuse the model — cover half their face, use dramatic lighting, hold up an ambiguous object, shoot from an extreme angle
   - The app shows the new classification results side-by-side with the original — highlighting how fragile the model's "understanding" is
   - A "confidence meter" visualizes how certain or uncertain the model is

5. **Step 4: "The algorithmic mirror"** (Section 4)
   - If a face is detected, the app applies a basic "beauty filter" using MediaPipe Face Mesh — smoothing, symmetry correction, feature resizing
   - The original and filtered faces are shown side-by-side
   - Facial landmarks are drawn over both versions so students can see exactly what was moved and by how much
   - A final prompt: *"Which one is you?"*

### Technical Stack

| Component | Technology | Why |
|---|---|---|
| Hosting | **GitHub Pages** | Free, static, no server needed, easy to share via QR code |
| Image classification | **TensorFlow.js + MobileNet** | Runs entirely in-browser, no API calls, fast on mobile |
| Face detection & mesh | **MediaPipe Face Mesh (JS)** | Client-side, real-time, gives 468 facial landmarks |
| Heatmap / attention | **tf-explain or custom GradCAM-lite** | Lightweight attention visualization |
| Beauty filter | **Canvas API + Face Mesh landmarks** | Simple geometric transforms using landmark coordinates |
| UI framework | **Vanilla JS or lightweight (e.g., Alpine.js)** | Minimal dependencies, fast load on mobile |
| Fallback for older devices | **Firebase Hosting** (if needed) | Only if GitHub Pages has CORS or performance issues with large WASM files |

### Why This Works for the Presentation

- **No friction:** Students scan a QR code and they're in. No app store, no login, no installation.
- **Privacy-first:** Everything runs on-device. No images are uploaded anywhere. This is itself a talking point — *most* CV tools don't work this way.
- **Spans all sections:** One tool, four lenses on the same image.
- **Photography-native:** Students are already thinking in images. This meets them where they are.
- **Keeps after the workshop:** The URL stays live. Students can revisit it, share it, use it as a reference.

### Repo Structure (suggested)

```
hkuworkshop/
├── index.html          # Main app entry point
├── css/
│   └── style.css
├── js/
│   ├── app.js          # Main app logic & step flow
│   ├── classifier.js   # MobileNet classification + GradCAM
│   ├── facemesh.js     # MediaPipe face detection & landmarks
│   └── filter.js       # Beauty filter transforms
├── assets/
│   └── ...             # Icons, fonts, example images
├── plan.md             # This file
└── README.md
```

---

## Key References

- Benjamin, Ruha. *Race After Technology: Abolitionist Tools for the New Jim Code* (2019)
- Crawford, Kate. *Atlas of AI: Power, Politics, and the Planetary Costs of Artificial Intelligence* (2021)
- Buolamwini, Joy & Gebru, Timnit. "Gender Shades" (2018)
- Paglen, Trevor & Crawford, Kate. "Excavating AI" (2019)
- Noble, Safiya Umoja. *Algorithms of Oppression* (2018)
- Foucault, Michel. *Discipline and Punish* (1975) — panopticism
- Steyerl, Hito. "Proxy Politics" (2014)
- Roth, Lorna. "Looking at Shirley, the Ultimate Norm" (2009)
