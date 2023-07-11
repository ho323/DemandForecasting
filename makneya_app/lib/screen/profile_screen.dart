import 'package:flutter/material.dart';
import 'package:syncfusion_flutter_charts/charts.dart';

class ProfileScreen extends StatefulWidget {
  const ProfileScreen({Key? key}) : super(key: key);

  @override
  State<ProfileScreen> createState() => _ProfileScreenState();
}

class _ProfileScreenState extends State<ProfileScreen> {
  @override
  Widget build(BuildContext context) {
    final List<ChartData> chartData = [
      ChartData('순수익', 26000000, Colors.blueAccent),
      ChartData('지출액', 72000000, Colors.redAccent),
    ];
    String monthSales = "98,000,000";

    return Scaffold(
      body: Padding(
        padding: const EdgeInsets.all(15.0),
        child: ListView(
          children: [
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceBetween,
              children: [
                Text(
                  "내 가게 정보",
                  style: TextStyle(
                    fontSize: 40.0,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                IconButton(
                  onPressed: () {},
                  icon: Icon(
                    Icons.settings_outlined,
                    size: 30.0,
                  ),
                ),
              ],
            ),
            Container(
              color: Colors.grey,
              height: 1.5,
              margin: EdgeInsets.symmetric(vertical: 20.0),
            ),
            Row(
              children: [
                Text(
                  "오늘",
                  style: TextStyle(
                    fontSize: 28.0,
                    fontWeight: FontWeight.w900,
                  ),
                ),
                PopupMenuButton(
                    child: Icon(
                      Icons.arrow_drop_down,
                      size: 40.0,
                    ),
                    itemBuilder: (BuildContext context) => []),
              ],
            ),
            Container(
              margin: EdgeInsets.all(20.0),
              child: Text(
                '"지출액"비율이 높아요!',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 22.0),
              ),
            ),
            SfCircularChart(
              series: <CircularSeries>[
                // Render pie chart
                PieSeries<ChartData, String>(
                  dataSource: chartData,
                  pointColorMapper: (ChartData data, _) => data.color,
                  xValueMapper: (ChartData data, _) => data.x,
                  yValueMapper: (ChartData data, _) => data.y,
                  explode: true,
                  explodeAll: true,
                  strokeColor: Colors.grey,
                  strokeWidth: 3.0,
                ),
              ],
            ),
            Text(
              "이번달 매출",
              style: TextStyle(
                fontSize: 16.0,
              ),
            ),
            Text(
              "$monthSales원",
              style: TextStyle(
                fontSize: 22.0,
                fontWeight: FontWeight.w700,
              ),
            ),
            SizedBox(height: 40),
            Text(
              "요약",
              style: TextStyle(
                fontSize: 16.0,
                fontWeight: FontWeight.w700,
              ),
            ),
            SizedBox(height: 8),
            salesSummary(month: 4, week: 4),
            salesSummary(month: 4, week: 3),
            salesSummary(month: 4, week: 2),
            salesSummary(month: 4, week: 1),
          ],
        ),
      ),
    );
  }

  Widget salesSummary({
    required int month,
    required int week,
  }) {
    return Container(
      margin: EdgeInsets.symmetric(vertical: 5.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Text(
                "$month월 $week주 ",
                style: TextStyle(
                  fontSize: 15.0,
                  fontWeight: FontWeight.w600,
                ),
              ),
              Text(
                "(04.23 - 04.29)",
                style: TextStyle(
                  color: Colors.grey,
                  fontSize: 13.0,
                  fontWeight: FontWeight.w600,
                ),
              ),
            ],
          ),
          Row(
            children: [
              Text(
                "28,000,000원",
                style: TextStyle(
                  fontSize: 15.0,
                  fontWeight: FontWeight.w700,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }
}

class ChartData {
  ChartData(this.x, this.y, [this.color]);
  final String x;
  final double y;
  final Color? color;
}
