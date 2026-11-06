use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

#[cfg(target_os = "macos")]
fn toolchain_swift_lib_dir() -> Option<String> {
    let output = Command::new("xcrun")
        .args(["--find", "swiftc"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }

    let swiftc = String::from_utf8(output.stdout).ok()?;
    let usr = Path::new(swiftc.trim()).parent()?.parent()?;
    Some(usr.join("lib/swift/macosx").to_string_lossy().into_owned())
}

fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rerun-if-changed=src/macos_audio.m");
        println!("cargo:rerun-if-changed=src/WispSpeech.swift");

        let out_dir = PathBuf::from(env::var("OUT_DIR").expect("OUT_DIR not set"));
        let obj_path = out_dir.join("macos_audio.o");
        let lib_path = out_dir.join("libmacos_audio.a");

        let status = Command::new("clang")
            .args([
                "-c",
                "-O2",
                "-fobjc-arc",
                "-Wall",
                "-Wextra",
                "-Wno-unused-parameter",
                "src/macos_audio.m",
                "-o",
            ])
            .arg(&obj_path)
            .status()
            .expect("Failed to execute clang to compile src/macos_audio.m");

        assert!(status.success(), "clang failed to compile src/macos_audio.m");

        let ar_status = Command::new("ar")
            .args(["crus"])
            .arg(&lib_path)
            .arg(&obj_path)
            .status()
            .expect("Failed to execute ar to archive libmacos_audio.a");

        assert!(ar_status.success(), "ar failed to create libmacos_audio.a");

        let swift_lib_path = out_dir.join("libWispSpeech.a");
        let swift_status = Command::new("swiftc")
            .args([
                "-emit-library",
                "-static",
                "-O",
                "-target",
                "arm64-apple-macosx11.0",
                "-module-name",
                "WispSpeech",
                "-o",
            ])
            .arg(&swift_lib_path)
            .arg("src/WispSpeech.swift")
            .status()
            .expect("Failed to execute swiftc to compile src/WispSpeech.swift");

        assert!(swift_status.success(), "swiftc failed to compile src/WispSpeech.swift");

        println!("cargo:rustc-link-search=native={}", out_dir.display());
        println!("cargo:rustc-link-lib=static=macos_audio");
        println!("cargo:rustc-link-lib=static=WispSpeech");

        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let lib_dir = manifest_dir.join("lib");
        println!("cargo:rustc-link-search=native={}", lib_dir.display());
        println!("cargo:rustc-link-lib=dylib=ParakeetBridge");
        println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_dir.display());
        println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");
        println!("cargo:rustc-link-arg=-Wl,-rpath,@executable_path");

        if let Some(swift_lib) = toolchain_swift_lib_dir() {
            println!("cargo:rustc-link-search=native={swift_lib}");
        }
        println!("cargo:rustc-link-arg=-Wl,-rpath,/usr/lib/swift");

        println!("cargo:rustc-link-lib=framework=ScreenCaptureKit");
        println!("cargo:rustc-link-lib=framework=SoundAnalysis");
        println!("cargo:rustc-link-lib=framework=AVFAudio");
        println!("cargo:rustc-link-lib=framework=AVFoundation");
        println!("cargo:rustc-link-lib=framework=CoreMedia");
        println!("cargo:rustc-link-lib=framework=CoreAudio");
        println!("cargo:rustc-link-lib=framework=CoreGraphics");
        println!("cargo:rustc-link-lib=framework=Foundation");
        println!("cargo:rustc-link-lib=framework=Speech");
        println!("cargo:rustc-link-lib=framework=CoreML");
        println!("cargo:rustc-link-lib=framework=Accelerate");

        let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
        let plist_path = manifest_dir.join("src/Info.plist");
        if plist_path.exists() {
            println!("cargo:rerun-if-changed=src/Info.plist");
            println!("cargo:rustc-link-arg=-sectcreate");
            println!("cargo:rustc-link-arg=__TEXT");
            println!("cargo:rustc-link-arg=__info_plist");
            println!("cargo:rustc-link-arg={}", plist_path.display());
        }
    }
}
