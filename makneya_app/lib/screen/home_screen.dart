import 'package:flutter/material.dart';
import 'package:lld/const/color.dart';
import 'package:lld/screen/item_screen.dart';
import 'package:lld/screen/menu_screen.dart';

class StoreList {
  static const store1 = "가게 1";
  static const store2 = "가게 2";
  static const store3 = "가게 3";
  static const List<String> menuItems = [store1, store2, store3];
}

class HomeScreen extends StatelessWidget {
  const HomeScreen({
    this.MoveSearch,
    this.MoveCash,
    this.MoveProfile,
    Key? key,
  }) : super(key: key);
  final MoveSearch;
  final MoveCash;
  final MoveProfile;

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: null,
      child: Scaffold(
        body: ListView(
          children: [
            renderSearch(),
            renderStoreList(),
            Row(
              children: [
                Expanded(child: renderMenu(context)),
                Expanded(child: renderItem(context)),
              ],
            ),
            renderCash(),
            Row(
              children: [
                Expanded(child: renderProfile()),
                Expanded(child: renderCheck()),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget renderSearch() {
    return GestureDetector(
      onTap: () {
        MoveSearch();
      },
      child: Container(
        height: 50,
        padding: EdgeInsets.symmetric(horizontal: 15.0),
        margin: EdgeInsets.all(15.0),
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
        child: Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Row(
              children: const [
                Text(
                  "검색",
                  style: TextStyle(color: Colors.grey, fontSize: 15.0),
                ),
              ],
            ),
            Row(
              children: const [
                Icon(
                  Icons.search,
                  color: Colors.grey,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget renderStoreList() {
    return Container(
      height: 50,
      margin: EdgeInsets.all(10.0),
      child: Row(
        children: [
          Text(
            " 내 가게",
            style: TextStyle(
              fontSize: 25.0,
              fontWeight: FontWeight.w900,
            ),
          ),
          PopupMenuButton(
            child: const Icon(Icons.arrow_drop_down, size: 40),
            itemBuilder: (BuildContext context) {
              return StoreList.menuItems.map((String item) {
                return PopupMenuItem(
                  value: item,
                  child: Text(item),
                );
              }).toList();
            },
          ),
        ],
      ),
    );
  }

  Widget renderMenu(context) {
    return GestureDetector(
      onTap: () {
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (BuildContext context) => MenuScreen(),
          ),
        );
      },
      child: Container(
        decoration: customBoxDecoration(),
        padding: EdgeInsets.all(8.0),
        margin: EdgeInsets.all(10.0),
        height: 180,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            Text(
              " 메뉴 관리",
              style: menuTextStyle(),
            ),
            Image(
              image: AssetImage('assets/pasta.png'),
              width: 115,
              height: 96,
            ),
          ],
        ),
      ),
    );
  }

  Widget renderItem(context) {
    return GestureDetector(
      onTap: () {
        Navigator.of(context).push(
          MaterialPageRoute(
            builder: (context) => ItemScreen(),
          ),
        );
      },
      child: Container(
        decoration: customBoxDecoration(),
        padding: EdgeInsets.all(8.0),
        margin: EdgeInsets.all(10.0),
        height: 180,
        child: Text(
          " 재고 관리",
          style: menuTextStyle(),
        ),
      ),
    );
  }

  Widget renderCash() {
    return GestureDetector(
      onTap: () {
        MoveCash();
      },
      child: Container(
        decoration: customBoxDecoration(),
        padding: EdgeInsets.all(8.0),
        margin: EdgeInsets.all(10.0),
        height: 180,
        child: Column(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              " 발주 정보",
              style: menuTextStyle(),
            ),
            Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  " 현재 잔액",
                  style: TextStyle(
                    color: Colors.grey,
                    fontSize: 22.0,
                  ),
                ),
                Row(
                  children: const [
                    Text(
                      " 13,400,000",
                      style: TextStyle(
                        fontSize: 35.0,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                    Text(
                      "₩",
                      style: TextStyle(
                        color: Colors.grey,
                        fontSize: 35.0,
                        fontWeight: FontWeight.w500,
                      ),
                    )
                  ],
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget renderProfile() {
    return GestureDetector(
      onTap: () {
        MoveProfile();
      },
      child: Container(
        decoration: customBoxDecoration(),
        padding: EdgeInsets.all(8.0),
        margin: EdgeInsets.all(10.0),
        height: 180,
        child: Text(
          " 매출 관리",
          style: menuTextStyle(),
        ),
      ),
    );
  }

  Widget renderCheck() {
    return GestureDetector(
      onTap: () {},
      child: Container(
        decoration: customBoxDecoration(),
        padding: EdgeInsets.all(8.0),
        margin: EdgeInsets.all(10.0),
        height: 180,
        child: Text(
          " 수령 품목 검수",
          style: menuTextStyle(),
        ),
      ),
    );
  }

  BoxDecoration customBoxDecoration() {
    return BoxDecoration(
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
    );
  }

  TextStyle menuTextStyle() {
    return const TextStyle(
      fontSize: 22.0,
      fontWeight: FontWeight.w900,
    );
  }
}
