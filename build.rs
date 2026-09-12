use std::{fs, path::PathBuf, process::Command};

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let output = Command::new(program).args(args).output().ok()?;
    output
        .status
        .success()
        .then(|| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
}

fn main() {
    let git_dir = fs::read_to_string(".git")
        .ok()
        .and_then(|dot_git| dot_git.trim().strip_prefix("gitdir: ").map(PathBuf::from))
        .or_else(|| {
            PathBuf::from(".git")
                .is_dir()
                .then(|| PathBuf::from(".git"))
        });
    if let Some(git_dir) = git_dir {
        println!("cargo:rerun-if-changed={}", git_dir.join("HEAD").display());
        let common_dir = fs::read_to_string(git_dir.join("commondir"))
            .ok()
            .map(|path| git_dir.join(path.trim()))
            .unwrap_or_else(|| git_dir.clone());
        println!(
            "cargo:rerun-if-changed={}",
            common_dir.join("packed-refs").display()
        );
        if let Ok(head) = fs::read_to_string(git_dir.join("HEAD")) {
            if let Some(reference) = head.trim().strip_prefix("ref: ") {
                println!(
                    "cargo:rerun-if-changed={}",
                    common_dir.join(reference).display()
                );
            }
        }
    } else {
        println!("cargo:rerun-if-changed=.git/HEAD");
    }

    let git_sha = command_output("git", &["rev-parse", "HEAD"]).unwrap_or_else(|| "unknown".into());
    let git_date = command_output("git", &["show", "-s", "--format=%cI", "HEAD"])
        .unwrap_or_else(|| "1970-01-01T00:00:00Z".into());
    let dirty = Command::new("git")
        .args(["status", "--porcelain", "--untracked-files=normal"])
        .output()
        .ok()
        .is_some_and(|output| output.status.success() && !output.stdout.is_empty());
    let rustc = command_output("rustc", &["--version"]).unwrap_or_else(|| "rustc unknown".into());

    println!("cargo:rustc-env=BRAINROUTER_GIT_SHA={git_sha}");
    println!("cargo:rustc-env=BRAINROUTER_GIT_COMMIT_DATE={git_date}");
    println!(
        "cargo:rustc-env=BRAINROUTER_GIT_DIRTY={}",
        if dirty { "true" } else { "false" }
    );
    println!("cargo:rustc-env=BRAINROUTER_RUSTC_VERSION={rustc}");
}
