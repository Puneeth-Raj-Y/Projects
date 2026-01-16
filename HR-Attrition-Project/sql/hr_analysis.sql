-- ===== STEP 1: CREATE DATABASE =====
DROP DATABASE IF EXISTS hr_project;
CREATE DATABASE hr_project;
USE hr_project;

-- ===== STEP 2: CREATE TABLE =====
CREATE TABLE hr_attrition (
EmpID INT,
Age INT,
Gender VARCHAR(10),
Department VARCHAR(50),
JobRole VARCHAR(50),
MonthlyIncome INT,
YearsAtCompany INT,
WorkLifeBalance INT,
OverTime VARCHAR(5),
JobSatisfaction INT,
PerformanceRating INT,
TrainingHours INT,
Attrition VARCHAR(5)
);

-- ===== STEP 3: INSERT SAMPLE DATA =====
INSERT INTO hr_attrition VALUES
(1001,34,'Male','IT','Data Analyst',65000,3,2,'Yes',3,4,12,'Yes'),
(1002,29,'Female','HR','HR Executive',42000,1,3,'No',2,3,8,'Yes'),
(1003,41,'Male','Finance','Accountant',72000,10,4,'No',4,5,15,'No'),
(1004,36,'Female','IT','Software Engineer',80000,7,3,'Yes',2,3,10,'Yes'),
(1005,27,'Male','Operations','Manager',50000,2,2,'Yes',1,2,6,'Yes'),
(1006,45,'Female','Finance','Senior Analyst',90000,15,4,'No',5,5,20,'No'),
(1007,31,'Male','HR','Recruiter',38000,1,1,'Yes',1,2,5,'Yes'),
(1008,39,'Female','IT','Team Lead',95000,9,4,'No',4,5,18,'No'),
(1009,28,'Male','Sales','Sales Exec',35000,1,2,'Yes',2,3,7,'Yes'),
(1010,44,'Female','Operations','Manager',78000,12,4,'No',4,5,16,'No');

-- ===== STEP 4: ANALYSIS QUERIES =====

-- 1) Attrition by Department
SELECT Department,
COUNT(*) AS Total_Employees,
SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS Employees_Left,
ROUND(100.0 * SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) / COUNT(*),2)
AS Attrition_Rate
FROM hr_attrition
GROUP BY Department;

-- 2) Income Band vs Attrition
SELECT 
CASE 
 WHEN MonthlyIncome < 50000 THEN 'Low'
 WHEN MonthlyIncome BETWEEN 50000 AND 80000 THEN 'Mid'
 ELSE 'High'
END AS Income_Band,

COUNT(*) AS Total_Employees,

SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS Employees_Left,

ROUND(100.0 * SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) / COUNT(*),2)
AS Attrition_Rate

FROM hr_attrition
GROUP BY Income_Band;

-- 3) Overtime Impact
SELECT 
OverTime,
COUNT(*) AS Total_Employees,
SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) AS Employees_Left,
ROUND(100.0 * SUM(CASE WHEN Attrition='Yes' THEN 1 ELSE 0 END) / COUNT(*),2)
AS Attrition_Rate
FROM hr_attrition
GROUP BY OverTime;
