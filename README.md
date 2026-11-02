# Instructor Effectiveness Modeling (EdTech Context)

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Format: Jupyter Notebook](https://img.shields.io/badge/Format-Jupyter%20Notebook-orange.svg)](https://jupyter.org/)

Technical assignment and end-to-end data science solution for the **Data Science / AI Content Specialist Intern** role. This project analyzes batch-level course data, defines a multi-dimensional instructor effectiveness score, aggregates batch metrics to the instructor level, and builds an interpretable Machine Learning classification model to predict instructor effectiveness tiers (`Low`, `Medium`, `High`).

---

## 📌 Project Context & Objectives

In an EdTech platform, the same course is delivered across multiple cohorts/batches by different instructors. An instructor may teach:
- Multiple batches of the same course
- The same course across different time periods
- Different courses across their tenure

Because student learning outcomes, engagement, and ratings fluctuate across batches, evaluating instructor quality requires more than simply averaging subjective feedback stars. 

### Core Goals:
1. **Define Instructor Effectiveness**: Formulate a justified, holistic composite metric balancing learner outcomes, engagement, and student feedback.
2. **Aggregate Batch-Level Data**: Design sound aggregation functions that account for sample size differences (instructors with 1–2 batches vs. 8+ batches).
3. **Build Classical ML Models**: Train classification models to predict effectiveness tiers while avoiding data leakage.
4. **Evaluate & Interpret**: Assess model trade-offs, highlight key drivers of instructional success, and explain real-world limitations.

---

## 📂 Repository Structure

```text
├── instructor_effectiveness_modeling.ipynb   # Main Jupyter Notebook (all code, visualizations & analysis)
├── instructor_data.csv                       # Batch-level EdTech dataset (208 batches, 60 instructors)
├── generate_data.py                          # Modular script to regenerate dataset matching handbook specs
├── requirements.txt                          # Python dependencies (strictly adhering to allowed libraries)
├── .gitignore                                # Git ignore file
└── README.md                                 # Complete project documentation & analysis answers
```

---

## 📊 Dataset Schema

Each row in `instructor_data.csv` represents a single course batch taught by an instructor:

| Category | Column | Description | Type / Range |
| :--- | :--- | :--- | :--- |
| **Identifiers** | `batch_id` | Unique ID for the course batch | String (`B_0001`...) |
| | `instructor_id` | Unique instructor identifier | String (`INST_001`...) |
| | `course_id` | Course identifier | String (`CRS_01`...) |
| **Learner Outcomes** | `completion_rate` | Fraction of learners completing the course | Float [0, 1] |
| | `dropout_rate` | Fraction of learners dropping out | Float [0, 1] |
| | `avg_score_improvement`| Average score gain from pre- to post-assessment | Continuous |
| | `avg_quiz_score` | Average quiz score achieved by the batch | Float [0, 100] |
| **Engagement** | `avg_watch_time` | Normalized average video watch time | Float [0, 1] |
| | `assignment_submission_rate`| Fraction of learners submitting assignments | Float [0, 1] |
| | `forum_activity_rate` | Fraction of learners participating in discussion forums | Float [0, 1] |
| **Feedback** | `avg_feedback_score` | Average learner rating for the instructor | Float [1, 5] |
| | `feedback_response_rate`| Fraction of learners who submitted feedback | Float [0, 1] |

---

## 🛠️ Methodology & Technical Architecture

### Step 1: Exploratory Data Analysis (EDA)
- Analyzed distributions for all outcome, engagement, and feedback metrics.
- Checked for missing values, outliers, and skewness.
- Plotted correlation matrices to uncover relationships (e.g., strong positive link between watch time and completion rate, and negative correlation between completion and dropout rate).
- Examined the distribution of batches taught per instructor, revealing significant variance (from 1 batch to 10 batches).

### Step 2: Defining Instructor Effectiveness
Instructor effectiveness is defined across three pillars:
$$\text{Score} = 0.40 \times \text{Outcomes} + 0.35 \times \text{Engagement} + 0.25 \times \text{Feedback}$$

1. **Learner Outcomes (40%)**:
   - `0.35 * completion_rate + 0.30 * norm(score_imp) + 0.25 * norm(quiz_score) - 0.10 * dropout_rate`
   - *Why*: Reflects actual knowledge acquisition and course completion.
2. **Engagement (35%)**:
   - `0.45 * watch_time + 0.40 * submission_rate + 0.15 * forum_activity`
   - *Why*: Shows whether the instructor sustains learner interest throughout the cohort.
3. **Learner Feedback (25%)**:
   - `norm(feedback_score) * (0.50 + 0.50 * feedback_response_rate)`
   - *Why*: Normalizes the 1–5 scale to [0, 1] and weights it by response rate so a 5.0 rating from only 5% of a class doesn't artificially inflate the score.

The raw score is min-max scaled to `[0, 1]` and divided into three balanced tiers:
- **Low**: Lowest 33% of instructors
- **Medium**: Middle 34% of instructors
- **High**: Top 33% of instructors

### Step 3: Batch-to-Instructor Level Aggregation
Since instructors teach varying numbers of batches:
- **Central Tendency (`mean`)**: Average performance across batches for all 9 metrics.
- **Consistency (`std`)**: Standard deviation across batches for completion and quiz scores. Single-batch instructors are assigned `0` (no observed variance yet).
- **Volume & Breadth**: `total_batches` and `distinct_courses` to account for instructor experience.

### Step 4: Machine Learning Model Training
We trained classical ML classifiers to predict the instructor's tier:
- **Algorithms**: Random Forest, Gradient Boosting, and Logistic Regression.
- **Feature Selection & Leakage Prevention**: The raw composite effectiveness score is strictly excluded from feature inputs. Features only include aggregated metrics, consistency indicators, and volume counts.
- **Validation**: 75/25 stratified train-test split + 5-fold stratified cross-validation.

### Step 5: Model Evaluation & Results
| Model | Test Accuracy | Macro F1-Score | 5-Fold Stratified CV Macro F1 |
| :--- | :---: | :---: | :---: |
| **Random Forest** | **0.800** | **0.780** | **0.771 (± 0.08)** |
| **Gradient Boosting** | 0.800 | 0.778 | 0.742 (± 0.09) |
| **Logistic Regression** | 0.733 | 0.718 | 0.725 (± 0.10) |

- **Class Imbalance**: Handled by setting equal tertile thresholds and using stratified splitting.
- **Error Distribution**: The confusion matrix shows that misclassifications only occur between adjacent classes (`Medium` $\leftrightarrow$ `High` or `Medium` $\leftrightarrow$ `Low`); there are zero extreme errors (`Low` $\leftrightarrow$ `High`).

---

## 💡 Mandatory Analysis Questions (Section 5)

### 1. Which features most influenced instructor effectiveness, and why?
Completion rate, assignment submission rate, video watch time, and average score improvement had the highest predictive importance in the model.
- **Reason**: In online and hybrid learning, student drop-off is the primary failure mode. Instructors who explain complex ideas clearly keep learners watching videos, submitting assignments on schedule, and completing assessments, which directly translates into demonstrable skill improvement.

### 2. Which variables could be misleading or confounded?
1. **Average feedback score**: Learner ratings suffer from popularity bias, leniency bias (easy graders receive higher marks), and self-selection bias due to low response rates.
2. **Course difficulty confounding**: Advanced technical subjects (e.g., Deep Learning, Distributed Systems) inherently suffer from lower completion rates and watch times than introductory courses, irrespective of instructor quality.
3. **Forum activity rate**: High forum traffic can indicate healthy peer discussions or severe student confusion caused by unclear lectures.

### 3. How could this model fail in real-world usage?
1. **Small sample size (Cold Start)**: For instructors who have only taught 1 or 2 batches, an unmotivated cohort can skew their aggregated metrics and wrongly label a good teacher as `Low`.
2. **Gaming the system**: If effectiveness metrics are known, instructors may make quizzes trivial or shorten assignments to artificially inflate completion and ratings.
3. **Curriculum & platform drift**: Changes to course structure, video length, or LMS platform directly shift baseline metrics, degrading model calibration.

### 4. What additional data would you want to improve this analysis?
1. **Student baseline prerequisites**: Pre-course diagnostic test scores and years of domain experience to control for incoming student ability.
2. **Course metadata**: Difficulty level, credit weight, syllabus depth, and delivery mode (live vs. recorded).
3. **Qualitative student feedback**: NLP topic modeling on text reviews to isolate teaching clarity from complaints about platform bugs or curriculum pacing.
4. **Long-term outcomes**: Capstone project completion, post-course certification clearance, and career placement rates.

### 5. Should this model be used for instructor performance evaluation? Why or why not?
**No, it should not be used for high-stakes evaluative decisions** (such as compensation, penalties, or termination).

- **Reason**:
  1. The model relies on observational correlation rather than causal attribution; it does not control for confounding factors like course difficulty and student baseline preparation.
  2. The sample size for new instructors (1–2 batches) is too small for statistically reliable classification.
  3. High-stakes usage incentivizes counterproductive behavior (lowering curriculum rigor).

**Recommended Use**: It should be deployed as an internal diagnostic tool to identify instructors who would benefit from pedagogical coaching, mentorship, or teaching support.

---

## 🚀 Quickstart & Setup

### 1. Clone the repository
```bash
git clone https://github.com/<your-username>/instructor-effectiveness-modeling.git
cd instructor-effectiveness-modeling
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run or explore the notebook
```bash
jupyter notebook instructor_effectiveness_modeling.ipynb
```
Or open directly in [Google Colab](https://colab.research.google.com/).

### 4. (Optional) Regenerate the dataset
```bash
python generate_data.py
```

---

## ⚖️ Guidelines & Constraints Compliance
- ❌ No LLMs or external APIs used in analysis or model code.
- ❌ No external datasets used.
- ❌ No AutoML tools used.
- ✅ Classical scikit-learn algorithms only.
- ✅ Completely reproducible random seed (`42`).
- ✅ All notebook cells pre-rendered with zero execution errors.

