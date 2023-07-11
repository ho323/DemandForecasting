import 'package:flutter/material.dart';
import 'package:lld/screen/login_register/logged_out_screen.dart';
import 'package:lld/screen/main_screen.dart';

class LoginScreen extends StatefulWidget {
  const LoginScreen({Key? key}) : super(key: key);

  @override
  State<LoginScreen> createState() => _LoginScreenState();
}

class _LoginScreenState extends State<LoginScreen> {
  TextEditingController controller = TextEditingController();
  TextEditingController controller2 = TextEditingController();

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.white,
      appBar: AppBar(
        title: Image(image: AssetImage('assets/makneya.png')),
        backgroundColor: Colors.white,
        elevation: 0.0,
        automaticallyImplyLeading: false,
        centerTitle: false,
      ),
      body: Padding(
        padding: const EdgeInsets.symmetric(vertical: 40.0, horizontal: 16.0),
        child: Column(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                IconButton(
                  onPressed: () {
                    Navigator.popUntil(
                      context,
                      ModalRoute.withName("/"),
                    );
                  },
                  icon: Icon(
                    Icons.keyboard_arrow_left,
                    size: 32.0,
                  ),
                ),
                Text(
                  "로그인",
                  style: TextStyle(fontSize: 24.0, fontWeight: FontWeight.w900),
                ),
                SizedBox(width: 40.0),
              ],
            ),
            SizedBox(height: 48.0),
            Form(
              child: Theme(
                data: ThemeData(
                  primaryColor: Colors.black,
                  inputDecorationTheme: InputDecorationTheme(
                    labelStyle: TextStyle(
                      color: Colors.black,
                      fontSize: 16.0,
                    ),
                  ),
                ),
                child: Column(
                  children: [
                    TextField(
                      controller: controller,
                      autofocus: true,
                      keyboardType: TextInputType.emailAddress,
                      decoration: InputDecoration(
                        hintText: "이메일 주소 입력",
                        contentPadding: const EdgeInsets.symmetric(
                            vertical: 10.0, horizontal: 20.0),
                        border: const OutlineInputBorder(
                          borderSide:
                              BorderSide(color: Colors.black, width: 2.0),
                        ),
                        enabledBorder: const OutlineInputBorder(
                          borderSide:
                              BorderSide(color: Colors.black, width: 1.0),
                        ),
                        focusedBorder: const OutlineInputBorder(
                          borderSide:
                              BorderSide(color: Colors.black, width: 2.0),
                        ),
                      ),
                    ),
                    SizedBox(height: 16.0),
                    TextField(
                      controller: controller2,
                      keyboardType: TextInputType.text,
                      obscureText: true,
                      decoration: InputDecoration(
                        hintText: "********",
                        contentPadding: const EdgeInsets.symmetric(
                            vertical: 10.0, horizontal: 20.0),
                        border: const OutlineInputBorder(
                          borderSide:
                              BorderSide(color: Colors.black, width: 2.0),
                        ),
                        enabledBorder: const OutlineInputBorder(
                          borderSide:
                              BorderSide(color: Colors.black, width: 1.0),
                        ),
                        focusedBorder: const OutlineInputBorder(
                          borderSide:
                              BorderSide(color: Colors.black, width: 2.0),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
            SizedBox(height: 16.0),
            GestureDetector(
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => MainScreen(),
                  ),
                );
              },
              child: Container(
                height: 52,
                decoration: BoxDecoration(
                  color: Colors.black,
                  borderRadius: BorderRadius.circular(10.0),
                  boxShadow: [
                    BoxShadow(
                      color: Colors.grey.withOpacity(0.5),
                      spreadRadius: 0,
                      blurRadius: 3,
                      offset: const Offset(0, 7),
                    ),
                  ],
                ),
                child: Center(
                  child: Text(
                    "LOG IN",
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                ),
              ),
            ),
            SizedBox(height: 24.0),
            Row(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                GestureDetector(
                  onTap: () {
                    print("object");
                  },
                  child: Text(
                    "아이디 찾기",
                    style: TextStyle(
                      color: Color(0xff787878),
                      fontSize: 14.0,
                    ),
                  ),
                ),
                Text(
                  " | ",
                  style: TextStyle(
                    color: Color(0xff787878),
                    fontSize: 14.0,
                  ),
                ),
                GestureDetector(
                  onTap: () {
                    print("object");
                  },
                  child: Text(
                    "비밀번호 찾기",
                    style: TextStyle(
                      color: Color(0xff787878),
                      fontSize: 14.0,
                    ),
                  ),
                ),
                Text(
                  " | ",
                  style: TextStyle(
                    color: Color(0xff787878),
                    fontSize: 14.0,
                  ),
                ),
                GestureDetector(
                  onTap: () {
                    print("object");
                  },
                  child: Text(
                    "회원가입",
                    style: TextStyle(
                      color: Color(0xff787878),
                      fontSize: 14.0,
                    ),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }
}
