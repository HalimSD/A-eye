from flask import Flask, render_template,request,flash,redirect,url_for,session
import sqlite3

app = Flask(__name__)
app.secret_key="123"

con=sqlite3.connect("database.db")
con.execute("create table if not exists user(pid integer primary key,name text,mail text)")
con.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login',methods=["GET","POST"])
def login():
    if request.method=='POST':
        name=request.form['name']
        password=request.form['password']
        con=sqlite3.connect("database.db")
        con.row_factory=sqlite3.Row
        cur=con.cursor()
        cur.execute("select * from user where name=? and mail=?",(name,password))
        data=cur.fetchone()

        if data:
            session["name"]=data["name"]
            session["mail"]=data["mail"]
            return redirect("user")
        else:
            flash("Username and Password Mismatch","danger")
    return redirect(url_for("index"))


@app.route('/user',methods=["GET","POST"])
def user():
    return render_template("user.html")

@app.route('/register',methods=['GET','POST'])
def register():
    if request.method=='POST':
        try:
            name=request.form['name']
            # address=request.form['address']
            # contact=request.form['contact']
            mail=request.form['mail']
            con=sqlite3.connect("database.db")
            cur=con.cursor()
            cur.execute("insert into user(name,mail)values(?,?)",(name,mail))
            con.commit()
            flash("Record Added  Successfully","success")
        except:
            flash("Error in Insert Operation","danger")
        finally:
            return redirect(url_for("index"))
            con.close()

    return render_template('register.html')

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for("index"))

if __name__ == '__main__':
    app.run(debug=True)
