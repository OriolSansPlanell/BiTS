import streamlit as st
from controllers.data_controller import DataController
from controllers.user_controller import UserController


def main():
    st.sidebar.title("Navigation")
    choice = st.sidebar.radio("Go to", ["Data Dashboard", "User Management"])

    if choice == "Data Dashboard":
        data_controller = DataController("data/sample.csv")
        data_controller.run()

    elif choice == "User Management":
        user_controller = UserController()
        user_controller.run()


if __name__ == "__main__":
    main()
