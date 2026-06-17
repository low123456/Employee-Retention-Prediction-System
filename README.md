# Employee Retention Prediction System

This project is a Streamlit web application that predicts employee retention risk from employee reviews. It combines review text, VADER sentiment analysis, employee ratings, and a trained hybrid CNN model to estimate the probability that an employee may leave the organization.

## Web Browser

The deployed application can be opened in a web browser here:

[Employee Retention Prediction System](https://employee-retention-prediction-system-uddnbmxssxsbwl9r9zvmeh.streamlit.app/)

## Features

- Employee login and account registration
- Employee review submission with department, rating, and work-life balance inputs
- Sentiment analysis using VADER
- Retention prediction using a trained hybrid CNN model
- Risk classification as Low, Medium, or High
- HR dashboard with review summaries, department filtering, charts, and high-risk employee records
- Supabase support for persistent cloud storage

## Main Technologies

- Python 3.11
- Streamlit
- TensorFlow / Keras
- scikit-learn
- pandas and NumPy
- VADER Sentiment
- Altair
- Supabase REST API

## Project Files

| File | Description |
| --- | --- |
| `app.py` | Main Streamlit application |
| `hybrid_cnn_model.keras` | Trained hybrid CNN retention model |
| `tokenizer.pkl` | Tokenizer used to convert review text into model input |
| `config.pkl` | Model configuration, including sequence length |
| `scaler.pkl` | Scaler for numeric rating features |
| `requirements.txt` | Python dependencies |
| `runtime.txt` | Python runtime version for Streamlit Cloud |
| `supabase_schema.sql` | Supabase database schema |
| `DEPLOYMENT.md` | Streamlit deployment instructions |

## User Roles

### Employee

Employees can log in, write a review, select their department, provide ratings, and submit the review for prediction. After submission, the system displays:

- Sentiment label
- Potential to leave
- Risk level

### HR

HR users can view the dashboard, filter reviews by department, and monitor:

- Total submitted reviews
- Average leave potential
- Number of high-risk employees
- Number of negative sentiment reviews
- Sentiment and risk charts
- Departments with higher leave potential
- Employees with high potential to leave

## Local Setup

1. Clone or download this project.
2. Make sure Python 3.11 is installed.
3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Make sure these model artifact files are available in the project folder:

```text
hybrid_cnn_model.keras
tokenizer.pkl
config.pkl
scaler.pkl
```

5. Run the application:

```bash
streamlit run app.py
```

6. Open the local URL shown by Streamlit in your web browser.

## Default Local Accounts

When running locally without Supabase data, the application creates default demo users:

| Role | Username | Password |
| --- | --- | --- |
| HR | `hr` | `admin123` |
| Employee | `employee` | `employee123` |

New accounts can also be created from the Register tab.

## Supabase Storage

The application can store users and employee reviews permanently in Supabase. To enable Supabase:

1. Create a Supabase project.
2. Run the SQL commands in `supabase_schema.sql`.
3. Add these secrets in Streamlit Cloud:

```toml
SUPABASE_URL = "https://your-project-id.supabase.co"
SUPABASE_ANON_KEY = "your-anon-public-key"
```

## Deployment

The project is designed for Streamlit Community Cloud. The deployed version is available at:

[https://employee-retention-prediction-system-uddnbmxssxsbwl9r9zvmeh.streamlit.app/](https://employee-retention-prediction-system-uddnbmxssxsbwl9r9zvmeh.streamlit.app/)

For deployment, upload the required app files and model artifacts to GitHub, connect the repository to Streamlit Cloud, set `app.py` as the main file, and configure Supabase secrets if persistent cloud storage is required.

## Notes

- The prediction is intended to support HR decision-making and should not be used as the only basis for employment decisions.
- Employee review text may contain sensitive information, so production deployments should use secure authentication and appropriate data access controls.
- Large training datasets, notebooks, duplicate models, and output folders are not required for deployment.
