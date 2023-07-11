import 'package:flutter/material.dart';

class CashScreen extends StatelessWidget {
  const CashScreen({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.symmetric(vertical: 20.0, horizontal: 16.0),
        child: ListView(
          children: [
            Text(
              "현재 잔액",
              style: TextStyle(
                color: Colors.grey,
                fontSize: 25.0,
              ),
            ),
            Row(
              children: [
                Text(
                  "13,400,000",
                  style: TextStyle(
                    fontSize: 40.0,
                  ),
                ),
                Text(
                  "₩",
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 40.0,
                  ),
                ),
              ],
            ),
            SizedBox(height: 25.0),
            Row(
              children: [
                Expanded(child: renderCard()),
                Expanded(child: renderAccount()),
              ],
            ),
            SizedBox(height: 25.0),
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  "발주",
                  style: TextStyle(
                    fontSize: 28.0,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                TextButton(
                  onPressed: () {},
                  child: Text(
                    "더보기",
                    style: TextStyle(
                      color: Colors.grey,
                      fontSize: 12.0,
                    ),
                  ),
                ),
              ],
            )
          ],
        ),
      ),
    );
  }

  Widget renderCard() {
    return GestureDetector(
      onTap: () {},
      child: Container(
        decoration: BoxDecoration(
          color: Colors.black,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.5),
              spreadRadius: 5,
              blurRadius: 10,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        padding: EdgeInsets.symmetric(vertical: 12.0, horizontal: 15.0),
        margin: EdgeInsets.fromLTRB(0, 10, 10, 10),
        height: 180,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "자동 충전 카드",
              style: CardTextStyle(),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    Text(
                      "Card",
                      style: CardTextStyle(),
                    ),
                    Text(
                      "HyundaiCard",
                      style: TextStyle(
                        color: Colors.white,
                        fontSize: 10.0,
                      ),
                    ),
                  ],
                ),
                SizedBox(height: 8.0),
                Text(
                  " Hyundai Zero Edition",
                  style: CardTextStyle(),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget renderAccount() {
    return GestureDetector(
      onTap: () {},
      child: Container(
        decoration: BoxDecoration(
          color: Colors.white,
          borderRadius: BorderRadius.circular(20),
          boxShadow: [
            BoxShadow(
              color: Colors.grey.withOpacity(0.5),
              spreadRadius: 5,
              blurRadius: 10,
              offset: const Offset(0, 3),
            ),
          ],
        ),
        padding: EdgeInsets.symmetric(vertical: 12.0, horizontal: 15.0),
        margin: EdgeInsets.fromLTRB(10, 10, 0, 10),
        height: 180,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              "연결 계좌 관리",
              style: AccountTextStyle(),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    SizedBox(),
                    Text(
                      "Shinhan Card ",
                      style: TextStyle(
                        color: Colors.blue,
                        fontSize: 11.0,
                        fontWeight: FontWeight.w700,
                      ),
                    ),
                  ],
                ),
                SizedBox(height: 8.0),
                Text(
                  "신한은행 00000000",
                  style: AccountTextStyle(),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  TextStyle CardTextStyle() {
    return TextStyle(
      color: Colors.white,
      fontSize: 15.0,
      fontWeight: FontWeight.w700,
    );
  }

  TextStyle AccountTextStyle() {
    return TextStyle(
      fontSize: 15.0,
      fontWeight: FontWeight.w700,
    );
  }
}
