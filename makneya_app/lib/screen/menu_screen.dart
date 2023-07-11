import 'package:flutter/material.dart';

class MenuScreen extends StatefulWidget {
  const MenuScreen({Key? key}) : super(key: key);

  @override
  State<MenuScreen> createState() => _MenuScreenState();
}

class _MenuScreenState extends State<MenuScreen> {
  bool menuSwitched = false;
  bool autoSwitched = false;

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: EdgeInsets.symmetric(
          vertical: 10.0,
        ),
        child: ListView(
          children: [
            renderTop(),
            SizedBox(height: 50.0),
            renderMenuCategory(),
            renderMenuItem(),
            renderMenuItem(),
            renderMenuItem(),
            renderMenuCategory(),
            renderMenuItem(),
            renderMenuItem(),
            SizedBox(height: 20.0),
            renderAutoFunction(),
          ],
        ),
      ),
    );
  }

  Widget renderTop() {
    return Container(
      margin: EdgeInsets.symmetric(
        vertical: 8.0,
        horizontal: 25.0,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            "메뉴 관리",
            style: TextStyle(fontSize: 25.0, fontWeight: FontWeight.w700),
          ),
          TextButton(
            onPressed: () {
              print("메뉴 설정 이동");
            },
            child: Text(
              "설정",
              style: TextStyle(
                color: Colors.grey,
                fontSize: 12.0,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget renderMenuCategory() {
    return Row(
      children: [
        Container(
          decoration: customBoxDecoration(),
          margin: EdgeInsets.symmetric(
            vertical: 8.0,
            horizontal: 25.0,
          ),
          padding: EdgeInsets.symmetric(
            vertical: 5.0,
            horizontal: 8.0,
          ),
          child: Text(
            "메뉴 카테고리",
            style: TextStyle(
              fontSize: 14.0,
              fontWeight: FontWeight.w700,
            ),
          ),
        ),
      ],
    );
  }

  Widget renderMenuItem() {
    return Container(
      decoration: customBoxDecoration(),
      margin: EdgeInsets.symmetric(
        vertical: 8.0,
        horizontal: 25.0,
      ),
      padding: EdgeInsets.symmetric(
        vertical: 8.0,
        horizontal: 16.0,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Icon(
                Icons.shopping_basket,
                color: Colors.red,
              ),
              SizedBox(width: 12.0),
              Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    "메뉴1",
                    style: TextStyle(
                      fontSize: 14.0,
                      fontWeight: FontWeight.w700,
                    ),
                  ),
                  Text(
                    "판매 가능 수량 : 0",
                    style: TextStyle(
                      color: Colors.grey,
                      fontSize: 14.0,
                    ),
                  ),
                ],
              ),
            ],
          ),
          Switch(
            value: menuSwitched,
            onChanged: (value) {
              setState(() {
                menuSwitched = value;
              });
            },
            activeColor: Colors.green,
          ),
        ],
      ),
    );
  }

  Widget renderAutoFunction() {
    return Container(
      decoration: customBoxDecoration(),
      margin: EdgeInsets.symmetric(
        vertical: 8.0,
        horizontal: 25.0,
      ),
      padding: EdgeInsets.symmetric(
        vertical: 8.0,
        horizontal: 16.0,
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Text(
                "자동 발주 기능",
                style: TextStyle(
                  fontSize: 15.0,
                  fontWeight: FontWeight.w900,
                ),
              ),
            ],
          ),
          Switch(
            value: autoSwitched,
            onChanged: (value) {
              setState(() {
                autoSwitched = value;
              });
            },
            activeColor: Colors.green,
          ),
        ],
      ),
    );
  }

  BoxDecoration customBoxDecoration() {
    return BoxDecoration(
      color: Colors.white,
      borderRadius: BorderRadius.circular(10),
      boxShadow: [
        BoxShadow(
          color: Colors.grey.withOpacity(0.5),
          spreadRadius: 2,
          blurRadius: 5,
          offset: const Offset(0, 3),
        ),
      ],
    );
  }
}
