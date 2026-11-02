import os
import numpy as np
import pandas as pd

def generate_instructor_dataset(
    output_path="instructor_data.csv",
    n_instructors=60,
    n_courses=8,
    seed=42
):
    """
    Generates a realistic batch-level dataset for EdTech instructor effectiveness modeling,
    strictly adhering to the schema and constraints outlined in Candidate Handbook.
    """
    np.random.seed(seed)
    
    instructors = [f"INST_{i:03d}" for i in range(1, n_instructors + 1)]
    courses = [f"CRS_{i:02d}" for i in range(1, n_courses + 1)]
    
    # Latent teaching capability distribution
    latent_skills = np.random.normal(0, 1, n_instructors)
    skill_map = dict(zip(instructors, latent_skills))
    
    records = []
    batch_counter = 1
    
    # Batch count distribution reflecting real-world EdTech workload variance
    batch_probs = [0.15, 0.20, 0.25, 0.15, 0.10, 0.08, 0.04, 0.03]
    batch_options = [1, 2, 3, 4, 5, 6, 8, 10]
    
    for inst in instructors:
        skill = skill_map[inst]
        n_batches = np.random.choice(batch_options, p=batch_probs)
        
        for _ in range(n_batches):
            course = np.random.choice(courses)
            
            # Outcome metrics
            completion = np.clip(0.65 + 0.12 * skill + np.random.normal(0, 0.08), 0.15, 0.98)
            dropout = np.clip(1.0 - completion + np.random.normal(0, 0.04), 0.02, 0.85)
            score_imp = np.clip(18.0 + 5.0 * skill + np.random.normal(0, 3.5), 2.0, 45.0)
            quiz_score = np.clip(72.0 + 6.0 * skill + np.random.normal(0, 4.0), 40.0, 98.0)
            
            # Engagement metrics
            watch_time = np.clip(0.60 + 0.10 * skill + np.random.normal(0, 0.09), 0.10, 0.95)
            submission_rate = np.clip(0.68 + 0.11 * skill + np.random.normal(0, 0.08), 0.20, 0.98)
            forum_rate = np.clip(0.35 + 0.08 * skill + np.random.normal(0, 0.10), 0.05, 0.90)
            
            # Feedback metrics
            feedback_score = np.clip(3.8 + 0.45 * skill + np.random.normal(0, 0.25), 1.5, 5.0)
            feedback_resp = np.clip(0.50 + 0.05 * skill + np.random.normal(0, 0.12), 0.10, 0.95)
            
            records.append({
                "batch_id": f"B_{batch_counter:04d}",
                "instructor_id": inst,
                "course_id": course,
                "completion_rate": round(float(completion), 3),
                "dropout_rate": round(float(dropout), 3),
                "avg_score_improvement": round(float(score_imp), 2),
                "avg_quiz_score": round(float(quiz_score), 2),
                "avg_watch_time": round(float(watch_time), 3),
                "assignment_submission_rate": round(float(submission_rate), 3),
                "forum_activity_rate": round(float(forum_rate), 3),
                "avg_feedback_score": round(float(feedback_score), 2),
                "feedback_response_rate": round(float(feedback_resp), 3)
            })
            batch_counter += 1
            
    df = pd.DataFrame(records)
    df.to_csv(output_path, index=False)
    print(f"Generated {len(df)} course batches across {len(instructors)} instructors -> saved to {output_path}")
    return df

if __name__ == "__main__":
    generate_instructor_dataset()

