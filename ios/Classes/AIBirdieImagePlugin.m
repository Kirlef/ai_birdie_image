#import "AIBirdieImagePlugin.h"
#if __has_include(<aibirdieimage/aibirdieimage-Swift.h>)
#import <aibirdieimage/aibirdieimage-Swift.h>
#else
// Support project import fallback if the generated compatibility header
// is not copied when this plugin is created as a library.
// https://forums.swift.org/t/swift-static-libraries-dont-copy-generated-objective-c-header/19816
#import "aibirdieimage-Swift.h"
#endif

@implementation AIBirdieImagePlugin
+ (void)registerWithRegistrar:(NSObject<FlutterPluginRegistrar>*)registrar {
  [SwiftAIBirdieImagePlugin registerWithRegistrar:registrar];
}
@end
