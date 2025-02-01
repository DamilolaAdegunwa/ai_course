import numpy as np
import pandas as pd
# Define the number of rows for the new dataset
num_rows = 750000


# Helper function to generate diverse full names
def generate_full_name():
    first_names = [
        "Oluwadamilola", "Pamela", "Zhang", "Juan", "Fatima", "Aarav", "Liam", "Sophia",
        "Abdullah", "Mei", "Akio", "Amina", "Thiago", "Elena", "Ibrahim", "Chidera",
        "Ahmed", "Mariam", "Hassan", "Rajesh", "Emily", "Olivia", "James", "Grace",
        "Peter", "Hua", "Kenji", "Fatou", "Carlos", "Lucia", "Ethan", "Isabella",
        "Noah", "Emma", "William", "Charlotte", "Benjamin", "Amara", "Omar", "Chika",
        "Laila", "Zahra", "Arjun", "Kiran", "Sara", "Aliyah", "Ayaan", "Rohan",
        "Daniel", "Hannah", "Jonathan", "Eva", "Miguel", "Sofia", "Dario", "Gabriela",
        "Victor", "Nina", "Enzo", "Isabel", "Matteo", "Chiara", "Francesco", "Giulia",
        "Aditya", "Lakshmi", "Vikram", "Priya", "Ananya", "Jayden", "Elijah", "Mia",
        "Aisha", "Amir", "Fatima", "Abigail", "Oscar", "Aurora", "Eleanor", "Eli",
        "Luna", "Zain", "Hailey", "Kofi", "Kwame", "Ama", "Yaw", "Akosua",
        "Tariq", "Malik", "Imani", "Jabari", "Nia", "Kwasi", "Akua", "Adwoa",
        "Derek", "Jessica", "Angel", "Brianna", "Vincent", "Kristina", "George", "Amanda",
        "Nathan", "Ashley", "Sergio", "Kimberly", "Antonio", "Maria", "Javier", "Carmen",
        "Luis", "Rosa", "Diego", "Luisa", "Marco", "Valeria", "Andrea", "Camila",
        "David", "Patricia", "Simon", "Theresa", "Klaus", "Stefanie", "Yuki", "Naoko",
        "Haruto", "Sakura", "Takeshi", "Akira", "Hiroshi", "Rika", "Ryo", "Aya",
        # Add more names to reach ~1000
    ]
    last_names = [
        "Adegunwa", "Kayiranga", "Lee", "Gonzalez", "Mohammed", "Sharma", "Smith", "Johnson",
        "Al-Farsi", "Chen", "Tanaka", "Ahmed", "Silva", "Ivanov", "Ali", "Nwankwo",
        "Williams", "Brown", "Jones", "Garcia", "Martinez", "Hernandez", "Lopez", "Wilson",
        "Anderson", "Thomas", "Moore", "Taylor", "Martin", "Jackson", "White", "Harris",
        "Sanchez", "Clark", "Lewis", "Robinson", "Walker", "Perez", "Hall", "Young",
        "Allen", "King", "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores",
        "Green", "Adams", "Nelson", "Baker", "Rivera", "Campbell", "Mitchell", "Carter",
        "Roberts", "Gomez", "Phillips", "Evans", "Turner", "Diaz", "Parker", "Cruz",
        "Edwards", "Collins", "Reyes", "Stewart", "Morris", "Morales", "Murphy", "Cook",
        "Rogers", "Gutierrez", "Ortiz", "Morgan", "Cooper", "Peterson", "Bailey", "Reed",
        "Kelly", "Howard", "Ramos", "Kim", "Cox", "Ward", "Richardson", "Watson",
        "Brooks", "Chavez", "Wood", "James", "Bennett", "Gray", "Mendoza", "Ruiz",
        "Hughes", "Price", "Alvarez", "Castillo", "Sanders", "Patel", "Myers", "Long",
        "Ross", "Foster", "Powell", "Jenkins", "Perry", "Russell", "Sullivan", "Bell",
        "Coleman", "Butler", "Henderson", "Barnes", "Gonzales", "Fisher", "Vasquez", "Simmons",
        "Romero", "Jordan", "Patterson", "Alexander", "Hamilton", "Graham", "Reynolds", "Griffin",
        # Add more unique names from diverse cultures (e.g., African, European, Asian, etc.)
        "Okonkwo", "Abdullahi", "Oluwaseun", "Fernandez", "Yamamoto", "Alonso", "Mejia", "Gupta",
        "Singh", "Khan", "Yusuf", "Hassan", "Zhang", "Wang", "Li", "Liu", "Kimura",
        "Takashi", "Kobayashi", "Chang", "Reddy", "Naidu", "Chaturvedi", "Bose", "Mukherjee",
        "O'Brien", "Doyle", "Murphy", "O'Connor", "McCarthy", "Fitzgerald", "Nielsen", "Hansen",
        "Larsen", "Jensen", "Kristensen", "Johansson", "Berg", "Lindberg", "Gustafsson", "Eriksson",
        "Schmidt", "Weber", "Müller", "Fischer", "Becker", "Hoffmann", "Schneider", "Meier",
        "Novak", "Svoboda", "Vesely", "Horak", "Kovacs", "Toth", "Balogh", "Molnar",
        "Popescu", "Ionescu", "Dumitru", "Stan", "Radulescu", "Ivanov", "Kirilov", "Popov",
        # Repeat and shuffle to reach 1,000
    ]
    other_names = [
        "Moyo", "Faith", "Wei", "Maria", "Hassan", "Raj", "James", "Grace",
        "Peter", "Hua", "Kenji", "Fatou", "Lima", "Petrov", "Omar", "Chioma",
        "Elena", "Ahmed", "Yuki", "Daniel", "Lucas", "Nina", "Liam", "Sophia",
        "Isabel", "Diego", "Thiago", "Sofia", "Aarav", "Olivia", "Emma", "Mia",
        "Amara", "Chidera", "Sara", "Aisha", "Rohan", "Amina", "Mei", "Juan",
        "Javier", "Ivan", "Carlos", "Luisa", "Hiroshi", "Ananya", "Vikram", "Priya",
        "Gabriela", "Camila", "Rosa", "Carmen", "Antonio", "Fatima", "Arjun", "Kiran",
        "Kwame", "Akosua", "Yaw", "Akua", "Nia", "Kwasi", "Adwoa", "Ama",
        "Imani", "Tariq", "Malik", "Jabari", "Oluwaseun", "Chika", "Laila", "Omar",
        "Fatoumata", "Kenjiro", "Mariama", "Amadou", "Abdoulaye", "Koffi", "Tunde", "Femi",
        "Ayodele", "Bolanle", "Abubakar", "Zainab", "Hajara", "Chinelo", "Ifeanyi", "Nkechi",
        "Takashi", "Sakura", "Naoko", "Aya", "Haruto", "Rika", "Ryo", "Kenichi",    "Zara",
        "Oluwadamilola", "Abdulrahman", "Ebele", "Chukwudi", "Adebayo", "Farida", "Salman",
        "Hamid", "Nafisa", "Gbenga", "Folake", "Adetola", "Babatunde", "Maimuna", "Sadiq",
        "Rasheed", "Funmilayo", "Yetunde", "Nnamdi", "Fadeke", "Balogun", "Olabisi", "Halima",
        "Sherif", "Bashir", "Taiwo", "Ademola", "Sekinat", "Bisi", "Tolulope", "Simisola",
        "Omotola", "Seydou", "Mamadou", "Fanta", "Kadija", "Ibrahim", "Mohammed", "Adekunle",
        "Olumide", "Bolanle", "Damilola", "Tolani", "Ayobami", "Ejiro", "Ireti", "Onome",
        "Alero", "Tamuno", "Amaka", "Obinna", "Chinyere", "Uche", "Ngozi", "Ifunanya",
        "Nkechi", "Ogechi", "Kelechi", "Somtochukwu", "Chizoba", "Adaeze", "Chioma", "Nkiruka",
        "Chisom", "Chibuzor", "Ifeoma", "Nonyelum", "Oluchi", "Zainab", "Basirat", "Latifat",
        "Omowunmi", "Aanuoluwapo", "Titilayo", "Folusho", "Seyi", "Adenike", "Omolabake", "Bosede",
        "Ruqayyah", "Abdulkareem", "Nurudeen", "Adebisi", "Tolu", "Afolabi", "Adeola", "Oladimeji",
        "Kudirat", "Isiaq", "Morayo", "Adijat", "Oladipo", "Abayomi", "Olajumoke", "Gbemi",
        "Ayoade", "Oreoluwa", "Taiye", "Kikelomo", "Olubunmi", "Bolanle", "Funsho", "Modupe"
        # Add more to reach ~1,000 unique names
    ]
    return f"{np.random.choice(first_names)} {np.random.choice(last_names)} {np.random.choice(other_names)}"


# Generate the new dataset
new_dataset = pd.DataFrame({
    "Name": [generate_full_name() for _ in range(num_rows)],
    "Age": np.random.randint(18, 66, size=num_rows),  # Age between 18 and 65
    "Salary": np.random.normal(60000, 15000, num_rows).astype(int),  # Salary with mean 60k, std 15k
    "Department": np.random.choice(
        ["Finance", "IT", "Marketing", "Operations", "Engineering", "HR"],
        size=num_rows,
        p=[0.2, 0.2, 0.15, 0.15, 0.2, 0.1]  # Approximate normal distribution
    ),
    "Join_Date": pd.to_datetime(np.random.choice(
        pd.date_range(start="1980-01-01", end="2023-01-01"),
        size=num_rows
    )),
    "Performance_Score": np.round(np.random.uniform(1, 5, num_rows), 2)  # Scores between 1.0 and 5.0
})

# Save the dataset to an Excel file
output_path = "750000_employee_dataset.xlsx"
new_dataset.to_excel(output_path, index=False)

#---
#---
#---
#---
