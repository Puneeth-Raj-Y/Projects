package com.company.main;

import java.util.List;
import java.util.Scanner;

import com.company.dto.EmployeeDTO;
import com.company.service.EmployeeService;

public class EmployeeApp {

    public static void main(String[] args) {

        Scanner sc = new Scanner(System.in);
        EmployeeService service = new EmployeeService();

        while (true) {

            System.out.println("\n===== EMPLOYEE MANAGEMENT SYSTEM =====");
            System.out.println("1. Add Employee");
            System.out.println("2. View All Employees");
            System.out.println("3. Search Employee by ID");
            System.out.println("4. Update Salary");
            System.out.println("5. Delete Employee");
            System.out.println("6. Exit");
            System.out.print("Enter choice: ");

            int choice = sc.nextInt();

            switch (choice) {

            case 1:
                System.out.print("Enter ID: ");
                int id = sc.nextInt();
                sc.nextLine();

                // 🔥 Check duplicate IMMEDIATELY after entering ID
                if (service.getEmployeeById(id) != null) {
                    System.out.println("Error: Employee ID already exists! Enter a different ID.");
                    break;   // stop here and return to menu
                }

                System.out.print("Enter Name: ");
                String name = sc.nextLine();

                System.out.print("Enter Department: ");
                String dept = sc.nextLine();

                System.out.print("Enter Salary: ");
                double salary = sc.nextDouble();
                sc.nextLine();

                System.out.print("Enter City: ");
                String city = sc.nextLine();

                EmployeeDTO emp = new EmployeeDTO(id, name, dept, salary, city);
                service.addEmployee(emp);
                break;


                case 2:
                    List<EmployeeDTO> list = service.getAllEmployees();
                    
                    System.out.println("\nEMPLOYEE_ID | NAME | DEPARTMENT | SALARY | CITY ");
                    System.out.println("--------------------------------------------------------------------");
                    for (EmployeeDTO e : list) {
                        System.out.println(
                                e.getEmpId() + " | " +
                                e.getName() + " | " +
                                e.getDepartment() + " | " +
                                e.getSalary() + " | " +
                                e.getCity()
                        );
                    }
                    break;

                case 3:
                    System.out.print("Enter Employee ID: ");
                    int searchId = sc.nextInt();
                    EmployeeDTO e = service.getEmployeeById(searchId);

                    if (e != null) {
                        System.out.println("Found: " + e.getName() + " | " + e.getSalary());
                    } else {
                        System.out.println("Employee not found!");
                    }
                    break;

                case 4:
                    System.out.print("Enter Employee ID: ");
                    int uid = sc.nextInt();

                    System.out.print("Enter New Salary: ");
                    double newSal = sc.nextDouble();

                    service.updateSalary(uid, newSal);
                    break;

                case 5:
                    System.out.print("Enter Employee ID to delete: ");
                    int delId = sc.nextInt();
                    service.deleteEmployee(delId);
                    break;

                case 6:
                    System.out.println("Goodbye!");
                    sc.close();
                    return;

                default:
                    System.out.println("Invalid choice!");
            }
        }
    }
}
