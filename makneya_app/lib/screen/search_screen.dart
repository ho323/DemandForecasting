import 'package:flutter/material.dart';

class SearchScreen extends StatelessWidget {
  const SearchScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: ListView(
        children: [
          Padding(
            padding: const EdgeInsets.all(15.0),
            child: Container(
              decoration: BoxDecoration(
                color: Colors.white,
                border: Border.all(
                  width: 3.0,
                  color: Colors.black,
                ),
                boxShadow: [
                  BoxShadow(
                    color: Colors.grey.withOpacity(0.5),
                    spreadRadius: 5,
                    blurRadius: 7,
                    offset: Offset(0, 3),
                  ),
                ],
              ),
              height: 50,
              padding: EdgeInsets.symmetric(horizontal: 15.0),
              child: TextFormField(
                decoration: InputDecoration(
                  hintText: "검색",
                  hintStyle: TextStyle(color: Colors.grey, fontSize: 15.0),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
