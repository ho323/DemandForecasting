import 'package:flutter/material.dart';
import 'package:lld/screen/login_register/login_screen.dart';

class RegisterCompleteScreen extends StatelessWidget {
  const RegisterCompleteScreen({Key? key}) : super(key: key);

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
        child: Container(
          width: double.infinity,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.center,
            children: [
              Text(
                "회원가입 완료",
                style: TextStyle(
                  fontSize: 25.0,
                  fontWeight: FontWeight.w900,
                ),
              ),
              SizedBox(height: 80.0),
              Image(image: AssetImage('assets/register_complete_icon.png')),
              SizedBox(height: 40.0),
              Text(
                "회원가입을 축하드립니다.",
                style: TextStyle(fontSize: 20.0),
              ),
              SizedBox(height: 20.0),
              Text("본인확인이 정상적으로 이루어져"),
              Text("회원가입이 완료 되었습니다."),
              SizedBox(height: 36.0),
              GestureDetector(
                onTap: () {
                  Navigator.push(
                    context,
                    MaterialPageRoute(
                      builder: (context) => LoginScreen(),
                    ),
                  );
                },
                child: Container(
                  height: 52,
                  decoration: BoxDecoration(
                    color: Color(0xFF000000),
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
            ],
          ),
        ),
      ),
    );
  }
}
