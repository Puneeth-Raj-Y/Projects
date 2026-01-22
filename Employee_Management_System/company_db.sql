CREATE DATABASE company_db;

USE company_db;

CREATE TABLE employees (
    emp_id INT PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    department VARCHAR(30),
    salary DOUBLE,
    city VARCHAR(30)
);

use company_db;
select * from employees;