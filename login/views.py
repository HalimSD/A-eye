from flask import Blueprint, render_template , Flask, request
from flask_login import login_required, current_user
import subprocess

views = Blueprint('views', __name__)


@views.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        if request.form.get("parms") == "cameracon":
            result = subprocess.check_output("python3 a_eye.py --project --conceptual", shell=True)
            print(result)
        elif request.form.get("parms") == "cameracoc":
            result = subprocess.check_output("python3 a_eye.py --project --coco", shell=True)
            print(result)
        elif request.form.get("parms") == "localcon":
            result = subprocess.check_output("python3 a_eye.py --project --conceptual", shell=True)
            print(result)
        elif request.form.get("parms") == "localcoc":
            result = subprocess.check_output("python3 a_eye.py --project --coco", shell=True)
            print(result)
    return render_template("home.html", user=current_user)
