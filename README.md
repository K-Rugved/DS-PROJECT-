# Experiment 7 – CI/CD Pipeline with Open Source Tools

## 🎯 Aim
To implement a CI/CD pipeline using open-source tools like GitHub Actions, DVC, Docker, and Python for automating testing, version checks, and deployment in a data science project.

---

## 🧪 Objective
- Automate testing and linting on every push to `main`
- Ensure model and code quality using CI tools
- Use DVC for dataset versioning
- Build and validate Docker image for deployment

---

## 🛠️ Tools Used
- GitHub Actions
- DVC (Data Version Control)
- Docker
- Python (pytest, flake8, ruff)

---

## 📁 Folder Structure

 ├── data/ # Dataset files (tracked with DVC)├── models/ # Trained model files 
 ├── src/ # Source code (pipeline, utils) 
 ├── tests/ # Unit tests 
 ├── workflows/ # GitHub Actions YAML files 
 ├── requirements.txt  # Python dependencies
 ├── README.md  # Project overview
 └── .gitignore


---

## ⚙️ CI/CD Workflow Summary

### ✅ `ci.yml` (in workflows/)
- Runs on push to `main`
- Sets up Python environment
- Installs dependencies
- Runs flake8 and ruff for linting
- Executes pytest for testing
- Optionally builds Docker image and runs health check

---

## 📦 DVC Integration
- Dataset tracked using DVC
- Remote storage configured
- `dvc pull` used in pipeline to fetch data

---

## 📸 Deliverables
- GitHub Actions workflow YAML file
- Screenshot of successful CI run
- Organized repo with proper folder structure
- Test file verifying model presence and loading


---


## ✅ Conclusion
Successfully implemented a CI/CD pipeline using GitHub Actions, DVC, and Docker to automate testing, linting, and deployment for a data science project. This setup ensures reproducibility, code quality, and smooth collaboration.


pytest
flake8
ruff
pytest-cov
fastapi
uvicorn