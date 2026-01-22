Employee Management System (Java + JDBC + MySQL)

📌 Project Overview

The **Employee Management System** is a console-based Java application that performs **CRUD operations** (Create, Read, Update, Delete) on employee records using **JDBC** and **MySQL**.
The project follows a **layered architecture (DAO + DTO + Service)** to ensure clean code separation, maintainability, and scalability.

🎯 Problem Statement

In many organizations, employee details are managed manually using spreadsheets or unstructured files. This leads to:

* Duplicate employee records
* Difficulty in searching and updating data
* Data inconsistency
* No centralized storage

This project solves these issues by storing employee data in a **relational database (MySQL)** and managing it using a structured Java application.

⚙️ Features

* Add new employee
* View all employees
* Search employee by ID
* Update employee salary
* Delete employee record
* Prevents duplicate employee IDs
* Validates salary and input data

---
🏗️ Project Architecture

The project follows **DAO + DTO + Service Layer Architecture**:

```
EmployeeApp (UI Layer)
        ↓
EmployeeService (Business Logic)
        ↓
EmployeeDAO / EmployeeDAOImpl (Database Logic)
        ↓
DBConnection (JDBC Utility)
        ↓
MySQL Database
```
Why this architecture?

* Separation of concerns
* Easy maintenance
* Better readability
* Industry-standard design

---
📂 Project Structure

```
Employee_Management_System
│
├── com.company.main
│   └── EmployeeApp.java
│
├── com.company.service
│   └── EmployeeService.java
│
├── com.company.dao
│   ├── EmployeeDAO.java
│   └── EmployeeDAOImpl.java
│
├── com.company.dto
│   └── EmployeeDTO.java
│
├── com.company.util
│   └── DBConnection.java
│
└── database
    └── employees table (MySQL)
```

---
🧾 Database Details

Database Name: `company_db`
Table Name: `employees`

```sql
CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(50),
    department VARCHAR(50),
    salary DOUBLE,
    city VARCHAR(50)
);
```

---

🛠️ Technologies Used

* Java
* JDBC
* MySQL
* Eclipse IDE
* MySQL Connector/J

---

▶️ How to Run the Project

1. Create database `company_db` in MySQL
2. Create the `employees` table
3. Update database credentials in `DBConnection.java`
4. Add **MySQL Connector/J** to the project build path
5. Run `EmployeeApp.java`

---
📌 Sample Console Menu

```
===== EMPLOYEE MANAGEMENT SYSTEM =====
1. Add Employee
2. View All Employees
3. Search Employee by ID
4. Update Salary
5. Delete Employee
6. Exit
```

---

✅ Key Learning Outcomes

* JDBC connectivity
* PreparedStatement usage
* Layered architecture implementation
* Clean separation of business logic and database logic
* Real-world CRUD operations

---
🚀 Conclusion

This project demonstrates how Java applications interact with databases using JDBC while following a professional layered architecture. It is suitable for **academic projects, interviews, and beginner backend development practice**.

---
