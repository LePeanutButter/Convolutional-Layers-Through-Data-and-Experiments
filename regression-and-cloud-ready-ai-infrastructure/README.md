# Stellar Luminosity Modeling from First Principles

**Escuela Colombiana de Ingeniería Julio Garavito**  
**Student:** Santiago Botero García

## Repository Structure and Contents

The repository contains the following files:

```
/
├── README.md
├── 01_part1_linreg_1feature.ipynb
└── 02_part2_polyreg.ipynb
```

### 01_part1_linreg_1feature.ipynb

This notebook implements **linear regression with a single feature** to model stellar luminosity as a function of stellar mass.

It includes:

- Dataset visualization and discussion of linearity
- Explicit definition of the hypothesis and loss function
- Cost surface visualization
- Analytical gradient derivation
- Gradient descent (non-vectorized and vectorized)
- Convergence analysis
- Experiments with multiple learning rates
- Final model fit and discussion of systematic errors
- Conceptual discussion of the astrophysical meaning and model limitations

All explanations and interpretations are documented directly within the notebook.

### 02_part2_polyreg.ipynb

This notebook extends the model to capture **nonlinear and interaction effects** using polynomial feature engineering.

It includes:

- Use of stellar mass and temperature as features
- Vectorized construction of the design matrix
- Fully vectorized loss and gradient computation
- Gradient descent training and convergence analysis
- Feature selection experiments comparing multiple models
- Analysis of the interaction term between mass and temperature
- An inference example for a new stellar configuration

All mathematical reasoning, results, and plots are explained inside the notebook.

## AWS SageMaker Execution Evidence

This section documents the execution of both notebooks in **AWS SageMaker**, following the official **Academy Learner Lab** setup instructions.

### SageMaker Setup (Academy Learner Lab)

The notebooks were executed using **Amazon SageMaker Studio**, respecting all Learner Lab constraints:

- IAM execution role: **LabRole**
- Supported instance types (e.g., `ml.t3.medium`)
- Maximum number of notebooks and applications as defined by the lab environment

#### SageMaker Domain Creation

A SageMaker Domain was created by following the Academy Learner Lab steps:

1. Created a new SageMaker Domain from the AWS Console.
2. Selected **Set up for organizations**.
3. Used an **existing IAM role (LabRole)** as the default execution role.
4. Configured applications:
   - SageMaker Studio
   - JupyterLab
   - Code Editor
   - Idle shutdown enabled (60 minutes).

5. Network configuration using the default VPC and public subnets.
6. Default security group selected.
7. Domain creation completed successfully.

#### User Profile and Studio Execution

- A **User Profile** was created and associated with the Domain.
- **SageMaker Studio** was launched using the created user profile.
- **JupyterLab** was opened from SageMaker Studio.
- Both notebooks were uploaded to JupyterLab and executed successfully.

SageMaker was used **only as an execution environment**.
No model deployment, endpoints, or MLOps pipelines were created.

### Execution Evidence Screenshots

The following screenshots are included as evidence of successful execution in SageMaker.

**1. Notebooks visible in SageMaker Studio**

![](<img/Screenshot 2026-01-26 210330.png>)

**2. Successful execution of all notebook cells (outputs visible)**

![](<img/Screenshot 2026-01-26 210343.png>)

**3. Example of a rendered plot inside SageMaker**

![](<img/Screenshot 2026-01-26 210356.png>)

### Local vs SageMaker Execution

Both local execution and AWS SageMaker execution produced identical numerical results and visualizations.

The main difference is that SageMaker provides a controlled, cloud-based environment aligned with enterprise architecture practices, without requiring local dependency management.

## Cost and Resource Management Notes

Following Academy Learner Lab recommendations:

- SageMaker Studio applications were stopped after use.
- No unnecessary instances were left running.
- No unsupported SageMaker features were used.
