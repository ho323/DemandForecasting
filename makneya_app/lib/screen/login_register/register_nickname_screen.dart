import 'package:flutter/material.dart';
import 'package:lld/screen/login_register/register_verification_screen.dart';

class RegisterNickNameScreen extends StatefulWidget {
  const RegisterNickNameScreen({Key? key}) : super(key: key);

  @override
  State<RegisterNickNameScreen> createState() => _RegisterNickNameScreenState();
}

class _RegisterNickNameScreenState extends State<RegisterNickNameScreen> {
  TextEditingController controller = TextEditingController();

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
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                // IconButton(
                //   onPressed: () {
                //     Navigator.pop(context);
                //   },
                //   icon: Icon(
                //     Icons.keyboard_arrow_left,
                //     size: 32.0,
                //   ),
                // ),
                Text(
                  "회원가입",
                  style: TextStyle(fontSize: 24.0, fontWeight: FontWeight.w900),
                ),
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
                child: TextField(
                  controller: controller,
                  autofocus: true,
                  keyboardType: TextInputType.emailAddress,
                  decoration: InputDecoration(
                    hintText: "이름 입력",
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
              ),
            ),
            SizedBox(height: 16.0),
            GestureDetector(
              onTap: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) => RegisterVerificationScreen(),
                  ),
                );
              },
              child: Container(
                height: 52,
                decoration: BoxDecoration(
                  color: Color(0xFFf0f0f0),
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
                    "NEXT",
                    style: TextStyle(
                      fontWeight: FontWeight.w900,
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
