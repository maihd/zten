const std = @import("std");

pub fn build(b: *std.build.Builder) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    const mode = b.standardReleaseOptions();

    const lib = createLibStep(b, mode);
    lib.install();

    const main_tests = b.addTest("src/main.zig");
    main_tests.setBuildMode(mode);
    linkLibC(main_tests);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&main_tests.step);
}

pub fn linkLibC(step: *std.build.LibExeObjStep) void {
    step.addIncludePath(thisDir() ++ "/libs/ggml/include");
    step.addIncludePath(thisDir() ++ "/libs/ggml/include/ggml");
    step.addCSourceFile(thisDir() ++ "/libs/ggml/src/ggml.c", &[_][]const u8 {
        "-DGGML_DEBUG=0",
    });
    step.linkLibC();
}

pub fn createLibStep(b: *std.build.Builder, mode: std.builtin.Mode) *std.build.LibExeObjStep {
    const lib = b.addStaticLibrary("zten", "src/main.zig");
    lib.setBuildMode(mode);

    linkLibC(lib);
    return lib;
}

pub fn link(b: *std.build.Builder, step: *std.build.LibExeObjStep, mode: std.builtin.Mode) void {
    const lib = createLibStep(b, mode);
    step.dependOn(&lib.step);
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}