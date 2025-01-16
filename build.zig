const std = @import("std");

pub fn build(b: *std.Build) void {
    // Standard release options allow the person running `zig build` to select
    // between Debug, ReleaseSafe, ReleaseFast, and ReleaseSmall.
    // const mode = b.standardReleaseOptions();

    const lib = createLibStep(b);
    b.installArtifact(lib);

    const main_tests = b.addTest(.{
        .root_source_file = b.path("src/main.zig")
    });
    // main_tests.setBuildMode(mode);
    linkLibC(b, main_tests);

    const test_step = b.step("test", "Run library tests");
    test_step.dependOn(&main_tests.step);
}

pub fn linkLibC(b: *std.Build, step: *std.Build.Step.Compile) void {
    step.addIncludePath(b.path(thisDir() ++ "/libs/ggml/include"));
    step.addIncludePath(b.path(thisDir() ++ "/libs/ggml/include/ggml"));
    step.addCSourceFile(.{
        .file = b.path(thisDir() ++ "/libs/ggml/src/ggml.c"), 
        .flags = &[_][]const u8 {
            "-DGGML_DEBUG=0",
        }
    });
    step.linkLibC();
}

pub fn createLibStep(b: *std.Build) *std.Build.Step.Compile {
    const lib = b.addStaticLibrary(.{
        .name = "zten",
        .root_source_file = b.path("src/main.zig"),
        .target = b.standardTargetOptions(.{})
    });

    linkLibC(b, lib);
    return lib;
}

pub fn link(b: *std.Build, step: *std.Build.Step.Compile) void {
    const lib = createLibStep(b);
    step.dependOn(&lib.step);
}

inline fn thisDir() []const u8 {
    return comptime std.fs.path.dirname(@src().file) orelse ".";
}