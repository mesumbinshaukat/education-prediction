def get_user_role(username):
    # You can use DB to fetch roles here
    if username == "admin":
        return "admin"
    return "student"
