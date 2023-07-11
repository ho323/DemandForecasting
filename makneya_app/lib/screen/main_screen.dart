import 'package:flutter/material.dart';
import 'package:lld/screen/cash_screen.dart';
import 'package:lld/screen/menu_screen.dart';
import 'package:lld/screen/profile_screen.dart';
import 'package:lld/screen/chat_screen.dart';
import 'package:lld/screen/home_screen.dart';
import 'package:lld/screen/search_screen.dart';

class MainScreen extends StatefulWidget {
  const MainScreen({Key? key}) : super(key: key);

  @override
  State<MainScreen> createState() => _MainScreenState();
}

class _MainScreenState extends State<MainScreen> {
  int tab = 0;

  MoveSearch() {
    setState(() {
      tab = 1;
    });
  }

  MoveCash() {
    setState(() {
      tab = 2;
    });
  }

  MoveProfile() {
    setState(() {
      tab = 4;
    });
  }

  @override
  Widget build(BuildContext context) {
    return WillPopScope(
      onWillPop: () {
        return Future(() => false);
      },
      child: Scaffold(
        backgroundColor: Colors.white,
        appBar: AppBar(
          title: Image(image: AssetImage('assets/makneya.png')),
          backgroundColor: Colors.white,
          elevation: 0.0,
          automaticallyImplyLeading: false,
          centerTitle: false,
          actions: [
            IconButton(
              onPressed: () {},
              icon: Icon(
                Icons.search,
                color: Colors.black,
                size: 30.0,
              ),
            ),
            IconButton(
              onPressed: () {},
              icon: Icon(
                Icons.notifications,
                color: Colors.black,
                size: 30.0,
              ),
            ),
            IconButton(
              onPressed: () {},
              icon: Icon(
                Icons.settings,
                color: Colors.black,
                size: 30.0,
              ),
            ),
          ],
        ),
        body: [
          HomeScreen(
              MoveSearch: MoveSearch,
              MoveProfile: MoveProfile,
              MoveCash: MoveCash),
          SearchScreen(),
          CashScreen(),
          ChatScreen(),
          ProfileScreen(),
        ][tab],
        bottomNavigationBar: BottomNavigationBar(
          showSelectedLabels: false,
          showUnselectedLabels: false,
          onTap: (i) {
            setState(() {
              tab = i;
            });
          },
          currentIndex: tab,
          items: const [
            BottomNavigationBarItem(
              icon: Icon(
                Icons.home_outlined,
                color: Colors.black,
              ),
              label: "home",
            ),
            BottomNavigationBarItem(
              icon: Icon(
                Icons.search,
                color: Colors.black,
              ),
              label: "search",
            ),
            BottomNavigationBarItem(
              icon: Icon(
                Icons.list_alt,
                color: Colors.black,
              ),
              label: "cash",
            ),
            BottomNavigationBarItem(
              icon: Icon(
                Icons.chat_bubble_outline,
                color: Colors.black,
              ),
              label: "chat",
            ),
            BottomNavigationBarItem(
              icon: Icon(
                Icons.person_outline,
                color: Colors.black,
              ),
              label: "profile",
            ),
          ],
        ),
      ),
    );
  }
}
