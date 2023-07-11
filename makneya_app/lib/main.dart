import 'package:flutter/material.dart';
import 'package:lld/screen/login_register/logged_out_screen.dart';
import 'package:lld/screen/main_screen.dart';

void main() {
  final bool isLoggined = false;
  runApp(
    MaterialApp(
      debugShowCheckedModeBanner: false,
      home: isLoggined ? MainScreen() : LoggedOutScreen(),
    ),
  );
}
