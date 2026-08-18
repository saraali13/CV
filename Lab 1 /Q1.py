# Task 1: Object-Oriented Grocery Manager

class GroceryManager:
    def __init__(self):
        self.grocery_list = {}

    def add_item(self, item, quantity, price):
        if quantity <= 0 or price < 0:
            print("Quantity must be positive and price cannot be negative.")
            return
        self.grocery_list[item] = {"quantity": quantity, "price": price}
        print(f"{item} added successfully.")

    def remove_item(self, item):
        if item in self.grocery_list:
            del self.grocery_list[item]
            print(f"{item} removed successfully.")
        else:
            print(f"Error: {item} does not exist in the list.")

    def view_list(self):
        if not self.grocery_list:
            print("The grocery list is empty.")
            return

        print("\nGrocery List")
        for item, details in self.grocery_list.items():
            quantity = details["quantity"]
            price = details["price"]
            print(f"{item}: Quantity = {quantity}, Price = {price:.2f}")

    def calculate_total(self):
        total = sum(
            details["quantity"] * details["price"]
            for details in self.grocery_list.values() )
        return total


manager = GroceryManager()
manager.add_item("Milk", 2, 250)
manager.add_item("Bread", 1, 180)
manager.add_item("Eggs", 12, 30)
manager.view_list()
print(f"Total cost: Rs. {manager.calculate_total():.2f}")
manager.remove_item("Bread")
manager.remove_item("Rice")

