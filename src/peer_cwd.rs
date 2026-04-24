//! Resolve the working directory of the process that owns a TCP peer socket.
//!
//! When OMP connects to brainrouter, each instance has a distinct cwd. By
//! mapping the peer's source port → client-side socket inode → owning pid →
//! /proc/{pid}/cwd we can surface the project folder in the dashboard without
//! any changes to OMP or its provider configuration.
//!
//! Only works on Linux (requires /proc/net/tcp and /proc/{pid}/fd).
//! Returns None silently on any I/O error so callers degrade gracefully.

use std::net::{IpAddr, SocketAddr};

/// Given the TCP peer address of an accepted connection, return the cwd of
/// the process that owns the client-side socket, or None if it cannot be
/// determined.
pub fn peer_cwd(peer_addr: &SocketAddr) -> Option<String> {
    let peer_port = peer_addr.port();

    let inode = match peer_addr.ip() {
        IpAddr::V4(_) => find_inode_for_client_port(peer_port, "/proc/net/tcp"),
        IpAddr::V6(_) => find_inode_for_client_port(peer_port, "/proc/net/tcp6"),
    }?;
    find_cwd_for_inode(inode)
}

/// Resolve the cwd of a process given its PID.
pub fn cwd_from_pid(pid: i32) -> Option<String> {
    let cwd_link = format!("/proc/{}/cwd", pid);
    if let Ok(cwd) = std::fs::read_link(&cwd_link) {
        return cwd.to_str().map(|s| s.trim_end_matches(" (deleted)").to_string());
    }
    None
}

/// Scan /proc/net/tcp or /proc/net/tcp6 for a row where local_port == client_port.
/// Returns the socket inode number.
fn find_inode_for_client_port(client_port: u16, path: &str) -> Option<u64> {
    let data = std::fs::read_to_string(path).ok()?;
    for line in data.lines().skip(1) {
        // Fields are whitespace-separated; split_ascii_whitespace handles variable spacing.
        let mut cols = line.split_ascii_whitespace();
        let _sl = cols.next()?;
        let local_addr = cols.next()?;  // "AABBCCDD:PPPP" or "AABBCCDDEEFF...:PPPP"
        let _rem_addr = cols.next()?;

        let local_port = local_addr
            .split(':')
            .nth(1)
            .and_then(|s| u16::from_str_radix(s, 16).ok())?;

        if local_port == client_port {
            // Skip: st tx_queue rx_queue tr tm->when retrnsmt uid timeout
            for _ in 0..7 { cols.next(); }
            let inode_str = cols.next()?;
            return inode_str.parse::<u64>().ok();
        }
    }
    None
}

/// Scan /proc/{pid}/fd for a symlink pointing to socket:[inode], then read
/// /proc/{pid}/cwd.
fn find_cwd_for_inode(inode: u64) -> Option<String> {
    let target = format!("socket:[{}]", inode);
    let proc = std::fs::read_dir("/proc").ok()?;

    for entry in proc.flatten() {
        let name = entry.file_name();
        let pid_str = match name.to_str() {
            Some(s) => s,
            None => continue,
        };
        if !pid_str.bytes().all(|b| b.is_ascii_digit()) {
            continue;
        }

        let fd_dir = format!("/proc/{}/fd", pid_str);
        let fds = match std::fs::read_dir(&fd_dir) {
            Ok(d) => d,
            Err(_) => continue, // no permission or process gone
        };

        for fd_entry in fds.flatten() {
            match std::fs::read_link(fd_entry.path()) {
                Ok(link) if link.to_str() == Some(&target) => {
                    // Found the process. Read its cwd.
                    let cwd_link = format!("/proc/{}/cwd", pid_str);
                    if let Ok(cwd) = std::fs::read_link(&cwd_link) {
                        return cwd.to_str().map(|s| s.trim_end_matches(" (deleted)").to_string());
                    }
                }
                _ => {}
            }
        }
    }
    None
}
