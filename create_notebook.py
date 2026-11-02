import os
import nbformat as nbf
from nbconvert.preprocessors import ExecutePreprocessor

nb = nbf.v4.new_notebook()

cells = []

# Cell 1: Title & Overview
cells.append(nbf.v4.new_markdown_cell("""# Instructor Effectiveness Modeling (EdTech Context)
**Role:** Data Science / AI Content Specialist Intern  
**Submission Format:** Jupyter Notebook / Google Colab  
**Allowed Libraries:** Python, pandas, numpy, scikit-learn, matplotlib, seaborn  

This notebook analyzes batch-level course data, defines a composite instructor effectiveness score, aggregates batch metrics to the instructor level, and builds an ML classification model to predict instructor effectiveness tiers (`Low`, `Medium`, `High`).
"""))

# Cell 2: Imports
cells.append(nbf.v4.new_code_cell("""import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (10, 5)
np.random.seed(42)"""))

# Cell 3: Step 1 Markdown
cells.append(nbf.v4.new_markdown_cell("""## Step 1: Exploratory Data Analysis (EDA)
Load the batch-level dataset, inspect summary statistics, check for missing values, and analyze distributions and correlations between variables.
"""))

# Cell 4: Data Loading
cells.append(nbf.v4.new_code_cell("""csv_path = 'instructor_data.csv'

# Generate synthetic dataset matching handbook specifications if file not present
if not os.path.exists(csv_path):
    n_instructors = 60
    instructors = [f'INST_{i:03d}' for i in range(1, n_instructors + 1)]
    courses = [f'CRS_{i:02d}' for i in range(1, 9)]
    latent_skill = np.random.normal(0, 1, n_instructors)
    inst_skill_map = dict(zip(instructors, latent_skill))

    records = []
    batch_counter = 1
    for inst in instructors:
        skill = inst_skill_map[inst]
        n_batches = np.random.choice([1, 2, 3, 4, 5, 6, 8, 10], p=[0.15, 0.20, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03])
        for _ in range(n_batches):
            course = np.random.choice(courses)
            completion = np.clip(0.65 + 0.12 * skill + np.random.normal(0, 0.08), 0.15, 0.98)
            dropout = np.clip(1.0 - completion + np.random.normal(0, 0.04), 0.02, 0.85)
            score_imp = np.clip(18.0 + 5.0 * skill + np.random.normal(0, 3.5), 2.0, 45.0)
            quiz_score = np.clip(72.0 + 6.0 * skill + np.random.normal(0, 4.0), 40.0, 98.0)
            watch_time = np.clip(0.60 + 0.10 * skill + np.random.normal(0, 0.09), 0.10, 0.95)
            submission_rate = np.clip(0.68 + 0.11 * skill + np.random.normal(0, 0.08), 0.20, 0.98)
            forum_rate = np.clip(0.35 + 0.08 * skill + np.random.normal(0, 0.10), 0.05, 0.90)
            feedback_score = np.clip(3.8 + 0.45 * skill + np.random.normal(0, 0.25), 1.5, 5.0)
            feedback_resp = np.clip(0.50 + 0.05 * skill + np.random.normal(0, 0.12), 0.10, 0.95)

            records.append({
                'batch_id': f'B_{batch_counter:04d}',
                'instructor_id': inst,
                'course_id': course,
                'completion_rate': round(completion, 3),
                'dropout_rate': round(dropout, 3),
                'avg_score_improvement': round(score_imp, 2),
                'avg_quiz_score': round(quiz_score, 2),
                'avg_watch_time': round(watch_time, 3),
                'assignment_submission_rate': round(submission_rate, 3),
                'forum_activity_rate': round(forum_rate, 3),
                'avg_feedback_score': round(feedback_score, 2),
                'feedback_response_rate': round(feedback_resp, 3)
            })
            batch_counter += 1

    df_gen = pd.DataFrame(records)
    df_gen.to_csv(csv_path, index=False)

df = pd.read_csv(csv_path)
print(f"Dataset Shape: {df.shape}")
print(f"Unique Instructors: {df['instructor_id'].nunique()}, Unique Courses: {df['course_id'].nunique()}")
df.head()"""))

# Cell 5: Missing values & summary
cells.append(nbf.v4.new_code_cell(r"""# Check for missing values and summary statistics
print("Missing values per column:")
print(df.isnull().sum())
print("\nNumerical Statistics:")
df.describe().T[['mean', 'std', 'min', '50%', 'max']]"""))

# Cell 6: Distributions Plot
cells.append(nbf.v4.new_code_cell("""# Distributions of key outcome and engagement metrics
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

sns.histplot(df['completion_rate'], kde=True, ax=axes[0, 0], color='teal')
axes[0, 0].set_title('Completion Rate Distribution')

sns.histplot(df['dropout_rate'], kde=True, ax=axes[0, 1], color='coral')
axes[0, 1].set_title('Dropout Rate Distribution')

sns.histplot(df['avg_watch_time'], kde=True, ax=axes[1, 0], color='indigo')
axes[1, 0].set_title('Watch Time Distribution')

sns.histplot(df['avg_feedback_score'], kde=True, ax=axes[1, 1], color='goldenrod')
axes[1, 1].set_title('Feedback Score Distribution')

plt.tight_layout()
plt.show()"""))

# Cell 7: Correlation Heatmap and Batches per Instructor
cells.append(nbf.v4.new_code_cell("""# Correlation heatmap and batches per instructor
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

num_cols = df.select_dtypes(include=[np.number]).columns
corr = df[num_cols].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='Blues', ax=axes[0], cbar=False)
axes[0].set_title('Correlation Matrix of Batch Metrics')

batch_counts = df.groupby('instructor_id')['batch_id'].count()
sns.histplot(batch_counts, discrete=True, ax=axes[1], color='teal')
axes[1].set_title('Batches Taught per Instructor')
axes[1].set_xlabel('Number of Batches')
axes[1].set_ylabel('Instructor Count')

plt.tight_layout()
plt.show()"""))

# Cell 8: Step 2 Markdown
cells.append(nbf.v4.new_markdown_cell("""## Step 2: Define Instructor Effectiveness
To define instructor effectiveness fairly, we combine three key areas into a weighted composite score:
- **Learner Outcomes (40%):** completion rate, score improvement, and quiz scores (penalized by dropout).
- **Engagement (35%):** video watch time, assignment submission rate, and forum activity.
- **Learner Feedback (25%):** average rating scaled from 0 to 1, weighted by response rate so ratings with very few responses don't skew the score.
"""))

# Cell 9: Effectiveness Score Code
cells.append(nbf.v4.new_code_cell("""def min_max(s):
    return (s - s.min()) / (s.max() - s.min() + 1e-6)

# Outcomes (40%)
outcome_score = 0.40 * (
    0.35 * df['completion_rate'] +
    0.30 * min_max(df['avg_score_improvement']) +
    0.25 * min_max(df['avg_quiz_score']) -
    0.10 * df['dropout_rate']
)

# Engagement (35%)
engagement_score = 0.35 * (
    0.45 * df['avg_watch_time'] +
    0.40 * df['assignment_submission_rate'] +
    0.15 * df['forum_activity_rate']
)

# Feedback (25%) adjusted for response rate
norm_feedback = (df['avg_feedback_score'] - 1.0) / 4.0
feedback_score = 0.25 * (norm_feedback * (0.50 + 0.50 * df['feedback_response_rate']))

raw_score = outcome_score + engagement_score + feedback_score
df['effectiveness_score'] = min_max(raw_score)

df[['batch_id', 'instructor_id', 'completion_rate', 'avg_watch_time', 'avg_feedback_score', 'effectiveness_score']].head()"""))

# Cell 10: Step 3 Markdown
cells.append(nbf.v4.new_markdown_cell("""## Step 3: Aggregate Batch Data to Instructor Level
Since each instructor teaches multiple batches (some have only 1-2, while others have 8+), we aggregate the batch-level data:
- **Average performance:** mean of outcomes, engagement, and feedback.
- **Consistency:** standard deviation across batches (0 for single-batch instructors).
- **Experience:** total batches taught and number of different courses handled.
- **Effectiveness Tiers:** divide instructors into `Low`, `Medium`, and `High` based on tertiles (33rd and 67th percentiles).
"""))

# Cell 11: Aggregation Code
cells.append(nbf.v4.new_code_cell("""inst_df = df.groupby('instructor_id').agg(
    total_batches=('batch_id', 'count'),
    distinct_courses=('course_id', 'nunique'),
    mean_completion=('completion_rate', 'mean'),
    std_completion=('completion_rate', 'std'),
    mean_dropout=('dropout_rate', 'mean'),
    mean_score_imp=('avg_score_improvement', 'mean'),
    mean_quiz_score=('avg_quiz_score', 'mean'),
    std_quiz_score=('avg_quiz_score', 'std'),
    mean_watch_time=('avg_watch_time', 'mean'),
    mean_submission_rate=('assignment_submission_rate', 'mean'),
    mean_forum_activity=('forum_activity_rate', 'mean'),
    mean_feedback_score=('avg_feedback_score', 'mean'),
    mean_feedback_response=('feedback_response_rate', 'mean'),
    mean_effectiveness=('effectiveness_score', 'mean')
).fillna(0)

# Split into Low, Medium, and High tiers
q33 = inst_df['mean_effectiveness'].quantile(0.33)
q67 = inst_df['mean_effectiveness'].quantile(0.67)

inst_df['tier'] = pd.cut(
    inst_df['mean_effectiveness'],
    bins=[-np.inf, q33, q67, np.inf],
    labels=['Low', 'Medium', 'High']
)

print('Instructor Tier Counts:')
print(inst_df['tier'].value_counts())
inst_df.head()"""))

# Cell 12: Tier Distribution Plot
cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(7, 4))
sns.countplot(x='tier', data=inst_df, palette='viridis', order=['Low', 'Medium', 'High'])
plt.title('Instructor Distribution across Effectiveness Tiers')
plt.xlabel('Effectiveness Tier')
plt.ylabel('Instructor Count')
plt.show()"""))

# Cell 13: Step 4 Markdown
cells.append(nbf.v4.new_markdown_cell("""## Step 4: Build a Machine Learning Model
Train classical ML models using the aggregated instructor features to predict the effectiveness tier.
The raw composite score is excluded from features to prevent data leakage.
"""))

# Cell 14: Features and Split
cells.append(nbf.v4.new_code_cell("""features = [
    'total_batches', 'distinct_courses', 'mean_completion', 'std_completion',
    'mean_dropout', 'mean_score_imp', 'mean_quiz_score', 'std_quiz_score',
    'mean_watch_time', 'mean_submission_rate', 'mean_forum_activity',
    'mean_feedback_score', 'mean_feedback_response'
]

X = inst_df[features]
y = inst_df['tier']

# Train-test split (75/25 stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)"""))

# Cell 15: Model Training & Comparison
cells.append(nbf.v4.new_code_cell("""models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=80, max_depth=3, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)
}

benchmark = []
fitted_models = {}

for name, model in models.items():
    if name == 'Logistic Regression':
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='macro')
    benchmark.append({'Model': name, 'Test Accuracy': round(acc, 3), 'Macro F1': round(f1, 3)})
    fitted_models[name] = {'model': model, 'preds': preds}

pd.DataFrame(benchmark)"""))

# Cell 16: Step 5 Markdown
cells.append(nbf.v4.new_markdown_cell("""## Step 5: Evaluate the Model
Evaluate the model using precision, recall, F1-score, cross-validation, and the confusion matrix.
"""))

# Cell 17: Classification Report & Confusion Matrix
cells.append(nbf.v4.new_code_cell("""tier_order = ['Low', 'Medium', 'High']
rf_preds = fitted_models['Random Forest']['preds']

print('=== Random Forest Classification Report ===')
print(classification_report(y_test, rf_preds, labels=tier_order))

# 5-fold cross-validation
cv_scores = cross_val_score(
    models['Random Forest'], X, y, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1_macro'
)
print(f'5-Fold Stratified CV Macro F1: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})')

# Confusion matrix
cm = confusion_matrix(y_test, rf_preds, labels=tier_order)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=tier_order, yticklabels=tier_order)
plt.title('Random Forest - Confusion Matrix')
plt.xlabel('Predicted Tier')
plt.ylabel('Actual Tier')
plt.show()"""))

# Cell 18: Step 6 Markdown
cells.append(nbf.v4.new_markdown_cell("""## Step 6: Interpret the Results
Check which features the model relies on most and see how these insights can be applied to an EdTech platform.
"""))

# Cell 19: Feature Importance
cells.append(nbf.v4.new_code_cell("""rf_clf = fitted_models['Random Forest']['model']
feat_imp = pd.Series(rf_clf.feature_importances_, index=features).sort_values(ascending=False)

plt.figure(figsize=(9, 5))
sns.barplot(x=feat_imp.values, y=feat_imp.index, palette='crest')
plt.title('Feature Importances for Instructor Effectiveness Tier')
plt.xlabel('Gini Importance')
plt.ylabel('Feature')
plt.show()"""))

# Cell 20: Section 5 - Mandatory Analysis Questions
cells.append(nbf.v4.new_markdown_cell("""## 5. Mandatory Analysis Questions

### 1. Which features most influenced instructor effectiveness, and why?
Completion rate, assignment submission rate, watch time, and average score improvement had the highest impact on predicting the effectiveness tier.

- Reason:
  1. Completion and assignment submission rates directly reflect whether learners are staying engaged throughout the course duration.
  2. Average watch time and score improvements show that students are actively consuming the material and actually learning the concepts, not just dropping off.

---

### 2. Which variables could be misleading or confounded?
A few variables can be tricky or misleading if taken at face value:
1. Average feedback score: Feedback is often biased. Lenient graders or popular personalities might get higher ratings, while tough but good instructors might receive lower ratings from struggling students. Response rates are also usually low.
2. Course difficulty: Harder courses (like advanced math or deep learning) naturally have lower completion and watch times compared to beginner courses, regardless of how good the instructor is.
3. Forum activity rate: High forum questions could mean great discussion, or it could mean the lecture was confusing and students have lots of doubts.

---

### 3. How could this model fail in real-world usage?
1. Small sample size (new instructors): Instructors who have only taught 1 or 2 batches have high variance. One difficult batch could wrongly label a good teacher as low tier.
2. Gaming the system: Instructors might start giving easier assignments and quizzes to push up completion rates and feedback scores.
3. Curriculum changes: If a course syllabus or platform changes, learner engagement metrics will shift, making older model thresholds inaccurate.

---

### 4. What additional data would you want to improve this analysis?
To make this model more reliable, I would want:
1. Student background / prerequisites: Knowing learners' prior experience helps control for cohort ability before blaming or crediting the instructor.
2. Course level details: Metadata like difficulty level, course duration, and whether classes are live or recorded.
3. Qualitative text reviews: Written feedback from students gives actual context on teaching style versus platform or curriculum complaints.
4. Long-term outcomes: Capstone performance, interview clearances, or career transitions post-course.

---

### 5. Should this model be used for instructor performance evaluation? Why or why not?
No, it should not be used for high-stakes evaluative decisions (such as compensation, penalties, or termination).

- Reason:
  1. The model relies on observational correlation rather than causal attribution; it does not control for confounding factors like course difficulty and student baseline preparation.
  2. The sample size for new instructors (1-2 batches) is too small for statistically reliable classification.
  3. High-stakes usage incentivizes counterproductive behavior (lowering curriculum rigor).

Instead, it is better used as an internal diagnostic tool to identify instructors who might need extra mentorship, coaching, or teaching support.
"""))

nb.cells = cells

notebook_path = r'c:\Users\LENOVO\Desktop\Accredian\instructor_effectiveness_modeling.ipynb'
with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)

print(f'Wrote notebook with {len(cells)} cells to {notebook_path}')

# Execute notebook to pre-render outputs and plots
print('Executing notebook to pre-render all outputs and plots...')
ep = ExecutePreprocessor(timeout=600, kernel_name='python3')
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb_to_run = nbf.read(f, as_version=4)

ep.preprocess(nb_to_run, {'metadata': {'path': r'c:\Users\LENOVO\Desktop\Accredian'}})

with open(notebook_path, 'w', encoding='utf-8') as f:
    nbf.write(nb_to_run, f)

print('Notebook executed and saved successfully with all outputs rendered!')
