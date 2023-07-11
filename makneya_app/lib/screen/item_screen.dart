import 'package:flutter/material.dart';

class ItemScreen extends StatelessWidget {
  const ItemScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      floatingActionButton: FloatingActionButton(
        onPressed: () {  },
        backgroundColor: Colors.black,
        child: Icon(Icons.add),
      ),
      body: Padding(
        padding: EdgeInsets.all(10.0),
        child: ListView(
          children: [
            Text(
              "재고 관리",
              style: TextStyle(fontSize: 25.0, fontWeight: FontWeight.w700),
            ),
            Container(
              margin: EdgeInsets.all(10.0),
              decoration: BoxDecoration(
                borderRadius: BorderRadius.circular(10),
                border: Border.all(
                  color: Colors.black,
                  width: 2.0,
                ),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.search,
                    color: Colors.grey,
                    size: 40.0,
                  ),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }
}
