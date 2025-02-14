class UserModel:
    def __init__(self):
        self.users = {}  # Simulated database

    def add_user(self, user_id, name):
        """Adds a new user."""
        self.users[user_id] = name

    def get_users(self):
        """Returns all users."""
        return self.users
