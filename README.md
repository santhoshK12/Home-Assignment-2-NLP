# Homework 2 — Natural Language Processing (CS5760)

**Student:** Santhosh Reddy Kistipati  
**Course:** CS5760 — NLP (Fall 2025)  
**University:** University of Central Missouri  
** Easily Accuss the all Data in the Github platform : https://github.com/santhoshK12/Home-Assignment-2-NLP/tree/main

This repository contains:
- **Theory answers (Q1–Q7)** with step-by-step explanations.
- **Programming implementations** for **Q5(c)** (metrics from confusion matrix) and **Q8** (Bigram Language Model).

---

##  Files

- `Homework 2 NLP(1-7).pdf` — Written theory answers for Q1–Q7.  
- `home_assignment_nlp(5th_and_8th).py` — Python script implementing Q5(c) & Q8.  
- `Home_Assignment_NLP(5th_and_8th).ipynb` — Jupyter notebook version of the same code.  
- `README.md` — (this file) question-by-question explanation and run instructions.

---

##  How to Run the Programming Parts

### Option A — Run the Python Script
```bash
python home_assignment_nlp(5th_and_8th).py
```
This prints:
- **Q5(c):** per-class precision/recall + macro & micro averages.
- **Q8:** unigram/bigram counts, step-by-step bigram probabilities for two sentences, final probabilities, and the model’s preference with a short explanation.

### Option B — Run the Notebook
Open `Home_Assignment_NLP(5th_and_8th).ipynb` in Jupyter/Colab and **Run All**.  
Outputs mirror the script.

---

## Q1. Bayes Rule Applied to Text

- **P(c):** prior probability of class *c* (e.g., how common “positive” vs “negative”).  
- **P(d|c):** likelihood of document *d* under class *c* (Naïve Bayes → product of word probabilities given the class).  
- **P(c|d):** posterior probability after observing document *d*.  
- **Why ignore P(d) when comparing classes:** for a fixed *d*, **P(d)** is constant across classes, so the argmax over classes depends only on **P(d|c)·P(c)**.

---

## Q2. Add‑1 (Laplace) Smoothing

Given: vocabulary size \(|V|=20\); negative class total tokens \(N_-=14\).  
- Denominator: \(N_- + |V| = 14 + 20 = \mathbf{34}\).  
- \(P(\text{predictable}\mid -) = (2+1)/34 = \mathbf{3/34} \approx 0.08824\).  
- \(P(\text{fun}\mid -) = (0+1)/34 = \mathbf{1/34} \approx 0.02941\).

---

## Q3. Worked Example — *“predictable no fun”*

Decision rule (multinomial NB):
\[
\text{score}(c) \propto P(c)\prod_{w\in d} P(w\mid c).
\]

**Negative class** (values known from Q2, plus \(P(\text{no}\mid -)\) by add‑1):  
\[
P(-)=\tfrac{3}{5},\quad 
P(\text{predictable}\mid -)=\tfrac{3}{34},\quad
P(\text{no}\mid -)=\tfrac{\text{cnt}_-(\text{no})+1}{34},\quad
P(\text{fun}\mid -)=\tfrac{1}{34}.
\]
\[
\text{score}(-)=\tfrac{3}{5}\cdot\tfrac{3}{34}\cdot\tfrac{\text{cnt}_-(\text{no})+1}{34}\cdot\tfrac{1}{34}.
\]

**Positive class** computed analogously using positive counts/total.  
**Assign** the label with the larger unnormalized score (or log‑score).

---

## Q4. Harms of Classification

- **Representational harm:** outputs that reinforce stereotypes or mischaracterize groups (e.g., neutral identity mentions rated as negative).  
- **Censorship risk:** toxicity filters can over‑block reclaimed/benign uses of identity words → silencing affected communities.  
- **Why worse on AAE/Indian English:** domain shift & data under‑representation → higher error due to lexical/syntactic differences and labeling bias.

---

## Q5. Evaluation Metrics from a Confusion Matrix

Confusion matrix (rows = **Predicted**, cols = **Gold**):

|            | Cat | Dog | Rabbit |
|------------|----:|----:|------:|
| **Cat**    |  5  | 10  |   5   |
| **Dog**    | 15  | 20  |  10   |
| **Rabbit** |  0  | 15  |  10   |

### Q5(a) Per‑class Precision & Recall
- **Cat:** Precision \(=5/20=0.25\), Recall \(=5/20=0.25\)  
- **Dog:** Precision \(=20/45\approx0.4444\), Recall \(=20/45\approx0.4444\)  
- **Rabbit:** Precision \(=10/25=0.4\), Recall \(=10/25=0.4\)

### Q5(b) Macro vs Micro
- **Macro (average over classes):** \((0.25+0.4444+0.4)/3=\mathbf{0.3648}\) for both precision & recall.  
- **Micro (aggregate over instances):** TP sum \(=5+20+10=35\); total \(=90\) → \(\mathbf{35/90=0.3889}\) for both precision & recall (equals accuracy in single‑label multi‑class).

### Q5(c) Programming (What the code does)
- **Input:** label list + confusion matrix.  
- **Per‑class:** compute row/column sums → \(TP,FP,FN\) → precision/recall per class.  
- **Macro:** mean of per‑class metrics.  
- **Micro:** \(\sum TP \,/\, \text{total}\) (precision=recall).  
- **Output:** prints per‑class, then macro & micro neatly with 4‑decimal formatting.

---

## Q6. Bigram Probabilities & Zero‑Probability

Training corpus:
```
<s> I love NLP </s>
<s> I love deep learning </s>
<s> deep learning is fun </s>
```

**S1 = `<s> I love NLP </s>`**  
\(P(I\mid<s>)=2/3\), \(P(love\mid I)=1\), \(P(NLP\mid love)=1/2\), \(P(</s>\mid NLP)=1\)  
\(\Rightarrow P(S1)=\tfrac{2}{3}\cdot1\cdot\tfrac{1}{2}\cdot1=\mathbf{1/3\approx0.3333}\).

**S2 = `<s> I love deep learning </s>`**  
\(P(I\mid<s>)=2/3\), \(P(love\mid I)=1\), \(P(deep\mid love)=1/2\), \(P(learning\mid deep)=1\), \(P(</s>\mid learning)=1/2\)  
\(\Rightarrow P(S2)=\tfrac{2}{3}\cdot1\cdot\tfrac{1}{2}\cdot1\cdot\tfrac{1}{2}=\mathbf{1/6\approx0.1667}\).

**Conclusion:** \(P(S1)>P(S2)\) → model prefers **S1**.

**Zero‑probability example:** after “ate” total=12 but count(noodle)=0 → \(P(\text{noodle}\mid ate)=0\).  
Problem: any sentence containing this bigram gets probability 0; perplexity becomes infinite.

**Laplace (Add‑1) smoothing:** with \(|V|=10\) and total after “ate”=12:  
\[
P(\text{noodle}\mid ate)=\frac{0+1}{12+10}=\frac{1}{22}\approx0.0455.
\]

---

## Q7. Backoff Model

Tiny corpus:
```
<s> I like cats </s>
<s> I like dogs </s>
<s> You like cats </s>
```
- \(P(\text{cats}\mid I,\!like)=\tfrac{\text{count}(I\,like\,cats)}{\text{count}(I\,like\,\cdot)}=\tfrac{1}{2}=0.5\).  
- Trigram \(P(\text{dogs}\mid You,\!like)\) unseen → **back off** to bigram:  
  \(P(\text{dogs}\mid like)=\tfrac{1}{3}\approx0.3333\).  
- **Why backoff:** trigrams are sparse; backoff uses broader context instead of assigning zero probability.

---

## Q8. Programming — Bigram Language Model (MLE)

**What the code does:**
1) **Read** the three training sentences (keeps `<s>` and `</s>`).  
2) **Tokenize** by whitespace.  
3) **Count** unigrams and bigrams; maintain `next_totals[w1] = Σ_x count(w1,x)`.  
4) **Estimate MLE** bigram probability: \(P(w_2\mid w_1)=\text{count}(w_1,w_2)/\sum_x\text{count}(w_1,x)\).  
5) **Sentence probability** = product of consecutive bigram probabilities; prints every step (`P(w_i|w_{i-1})`).  
6) **Test** two sentences: `<s> I love NLP </s>` and `<s> I love deep learning </s>`.  
7) **Print decision** with a short justification (S1 preferred since \(P(</s>\mid NLP)=1\) whereas \(P(</s>\mid learning)=1/2\)).

**Expected numeric results:**
- \(P(S1)=\mathbf{0.3333}\), \(P(S2)=\mathbf{0.1667}\) → **Prefer S1**.

---

##  Submission Checklist

- [x] Q1–Q7 theory: step-by-step in PDF.  
- [x] Q5(c) metrics program: script & notebook, prints per-class + macro/micro.  
- [x] Q8 bigram LM: script & notebook, prints counts, step-by-step probabilities, and preference.  
- [x] This README explains all Q1→Q8 and how to run code.  

---

## Reproducibility Tips

- **Q5(c):** To test another confusion matrix, edit `labels` and `confusion_matrix` at the top of the script.  
- **Q8:** To test new sentences, change `s1`/`s2` or call `sentence_probability("<s> ... </s>", ...)`.  
- For smoothing/backoff experiments, you can extend the code to add add‑1 smoothing or Katz/KN backoff.
