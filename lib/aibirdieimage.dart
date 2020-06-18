import 'dart:async';

import 'package:flutter/services.dart';

import 'src/classifier.dart';

class AIBirdieImage {
  static const MethodChannel _channel = const MethodChannel('aibirdieimage');

  static Classifier classification() => Classifier.instance;

  static Future<String> get platformVersion async {
    final String version = await _channel.invokeMethod('getPlatformVersion');
    return version;
  }
}
