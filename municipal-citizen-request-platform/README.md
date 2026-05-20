# Municipal Citizen Request Management Platform

**Escuela Colombiana de Ingeniería Julio Garavito**  
**Students:** Santiago Amaya Zapata, Andrés Ricardo Ayala Garzón & Santiago Botero García

## Executive Summary

This project presents a comprehensive technical, financial, and agile planning analysis for the development of a municipal citizen request management platform for Colombian municipalities. The platform serves as a digital nerve center connecting residents directly to municipal services through a unified, accessible system, replacing fragmented phone lines, paper records, and social media messages with organized digital service delivery.

**Key Outcomes:**

- **Financial Correction:** Identified and corrected 57.8% budget inflation from original international rates to realistic Colombian market costs
- **Velocity Alignment:** Established realistic 18-22 story points per sprint velocity for Colombian development teams
- **Technical Validation:** Comprehensive modular monolith architecture with full Colombian compliance integration
- **Labor Market Validation:** All salary and cost assumptions validated against Colombian market data with mandatory employer contributions

## System Overview

### Architecture

The platform employs a **modular monolith architecture** using React 18, Node.js (NestJS), PostgreSQL with PostGIS, and integrates with AWS S3, Twilio, and Auth0. This approach ensures manageable codebase while supporting future scalability for Colombian municipal growth.

### Core Modules

- **Citizen Portal:** Responsive web app with GPS/camera APIs and PWA offline capability
- **Admin Dashboard:** Data-rich interface for dispatchers and field crews
- **Backend Services:** API gateway, business logic layer, worker service, and audit logging
- **External Integrations:** Authentication, maps, notifications, and analytics services

### Technical Specifications

- **Security:** AES-256 encryption, OAuth 2.0, RBAC with Colombian municipal role hierarchy
- **Performance:** <2 second response times, 99.5% uptime, 10,000 concurrent user support
- **Data Sovereignty:** All citizen data stored within Colombian borders per government requirements

## Agile Planning Overview

### Sprint Structure

- **Duration:** 2-week sprints (10 working days)
- **Team:** 6 members (1 Frontend, 2 Backend, 1 QA, 1 PM/Scrum Master, 1 UI/UX Designer)
- **Total Story Points:** 89 across 5 sprints

### Velocity Assumptions (Corrected)

- **Target Range:** 18-22 points per sprint (realistic for Colombian mid-level teams)
- **Factors:** Colombian communication efficiency, technical density, compliance requirements
- **Monitoring:** Actual vs. planned variance tracking with quality metrics

### Story Point Distribution

- **Technical Foundation:** 20 points (22.5%)
- **User-Facing Features:** 69 points (77.5%)
- **Complexity:** Low (53.1%), Medium (37.5%), High (3.1%)
- **Priority:** Must-Have (65.2%), Should-Have (26.9%), Could-Have (7.9%)

## Financial Summary

### Original vs. Corrected Budget Comparison

| Component            | Original (USD) | Corrected (USD) | Difference   | % Change   |
| -------------------- | -------------- | --------------- | ------------ | ---------- |
| Development Cost     | $117,600       | $43,330         | -$74,270     | -63.2%     |
| Infrastructure       | $3,000         | $7,580          | +$4,580      | +152.7%    |
| **Total Base Cost**  | **$120,600**   | **$50,910**     | **-$69,690** | **-57.8%** |
| With 20% Contingency | $144,720       | $61,092         | -$83,628     | -57.8%     |

### Colombian Labor Cost Adjustments

- **Salaries:** Adjusted from international rates to Colombian market standards (Remoti, 2024)
- **Employer Contributions:** Added 31% mandatory social security and parafiscal contributions (RemoFirst, 2024)
- **Exchange Rate:** 1 USD = 3,560.62 COP (April 25, 2026)

### Final Validated Cost Estimate

- **Development Cost:** 154,325,000 COP ($43,330 USD)
- **Infrastructure & Services:** 27,000,000 COP ($7,580 USD)
- **Total with Contingency:** 217,590,000 COP ($61,092 USD)

## Key Findings from Audit

### Inconsistencies Found

1. **Budget Inflation:** Original estimates used international salary rates rather than Colombian market standards
2. **Missing Technical Tasks:** Initial backlog omitted critical foundation stories (database setup, authentication, infrastructure)
3. **Velocity Misalignment:** Original planning didn't account for Colombian team learning curves and compliance complexity

### Budget Inflation Issue

- **Root Cause:** Use of international salary benchmarks without Colombian market adjustment
- **Impact:** 57.8% cost inflation making project appear financially unviable
- **Resolution:** Applied Colombian market rates with mandatory employer contributions

### Velocity Misalignment

- **Issue:** Initial velocity estimates didn't consider Colombian municipal compliance requirements
- **Resolution:** Adjusted to 18-22 points per sprint based on local team capabilities

### Missing Technical Tasks

- **Gap:** Critical infrastructure and setup stories were omitted from initial scope
- **Resolution:** Added 20 additional story points for technical foundation work

### Salary Benchmarking Corrections

- **Method:** Validated against Remoti and RemoFirst Colombian tech salary databases
- **Result:** Realistic mid-level developer salaries with full benefits compliance

## Academic Paper Reference

A comprehensive academic LaTeX paper containing the complete technical, financial, and agile analysis is available in:

```
/paper/main.tex
```

The academic paper provides:

- **Formal Documentation:** Complete technical specifications, architecture diagrams, and implementation details
- **Financial Analysis:** Detailed cost breakdowns, budget validation, and ROI calculations
- **Agile Methodology:** Sprint planning, velocity calculations, and team composition analysis
- **Source Validation:** APA-style citations with full bibliography of 40+ verified sources
- **Colombian Context:** All assumptions validated against local regulations, labor laws, and market conditions

## Source Validation Summary

All numerical assumptions are backed by verified sources using APA-style citations:

### Reliability Approach

- **Economic Data:** World Bank, OECD, and Colombian government publications
- **Labor Market:** Remoti and RemoFirst HR platforms with Colombian specialization
- **Technical Standards:** Official documentation from technology providers (AWS, Auth0, React, etc.)
- **Government Regulations:** Colombian ICT Ministry, Data Protection Authority, and Labor Ministry
- **Industry Best Practices:** Atlassian, Microsoft, and consulting firm research

### Verification Methodology

- **Cross-Reference:** Multiple independent sources for each assumption
- **Currency Validation:** Exchange rates from official Colombian sources
- **Compliance Checking:** All requirements validated against Colombian law
- **Market Analysis:** Salary and cost data compared across multiple Colombian platforms

## Project Structure

```
/tdse-lab/
├── docs/
│   ├── 1-problem-scope.md
│   ├── 2-architecture.md
│   ├── 3-architecture-explanation.md
│   ├── 4-product-backlog.md
│   ├── 5-story-point.md
│   ├── 6-team-velocity.md
│   ├── 7-number-sprints.md
│   ├── 8-sprint-cost.md
│   ├── 9-total-budget.md
│   └── 10-table.md
├── paper/
│   └── main.tex
└── README.md
```

### Documentation Overview

- **`docs/` folder:** Contains 10 detailed markdown files with APA citations and source validation for each project aspect
- **`paper/main.tex`**: Complete academic paper consolidating all docs content into formal LaTeX format with tables, citations, and bibliography
- **`README.md`**: This file - unified project overview and entry point for all stakeholders

### Usage Guidelines

- **Technical Review:** See `paper/main.tex` for complete technical specifications and architecture details
- **Financial Analysis:** Refer to budget tables and cost validation in both docs files and academic paper
- **Implementation Planning:** Use sprint breakdowns and velocity calculations from agile planning sections
- **Source Verification:** All assumptions traceable to validated sources in individual docs files
