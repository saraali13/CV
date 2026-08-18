# Task 2: Advanced Student Record System

students = {
    "S001": {"Name": "Ali", "Major": "AI", "Grades": [85, 90, 88]},
    "S002": {"Name": "Sara", "Major": "AI", "Grades": [92, 95, 91]},
    "S003": {"Name": "Abbas", "Major": "CY", "Grades": [78, 82, 80]},
}


def highest_average_student(records):
    if not records:
        return None

    best_student = None
    highest_average = 0

    for student_id, data in records.items():
        grades = data["Grades"]

        if grades:
            average = sum(grades) / len(grades)

            if average > highest_average:
                highest_average = average
                best_student = data["Name"]

    return best_student


def search_by_major(records, major):
    found = False
    print(f"\nStudents in {major}:")

    for data in records.values():
        if data["Major"].lower() == major.lower():
            print(data["Name"])
            found = True

    if not found:
        print("No students found in this major.")


print("Student with highest average:", highest_average_student(students))
search_by_major(students, "AI")
