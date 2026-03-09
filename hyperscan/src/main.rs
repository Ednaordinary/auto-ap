use std::process::Command;
use std::sync::LazyLock;
use std::thread;
use std::time::Duration;

use regex::{Regex, RegexBuilder};

#[derive(Default)]
pub struct Wifi {
    pub mac: String,
    pub ssid: String,
    pub channel: String,
    pub signal: String,
    pub seen: String,
}


fn parse_scan(lines: &str) -> regex::Captures {
    static RE: LazyLock<Regex> = LazyLock::new(|| {
        RegexBuilder::new(r"BSS (?P<mac>.{17}?)\(.*?freq: (?P<freq>.+?)\n.*?signal: (?P<signal>.+?) .*?last seen: (?P<seen>.*?)\n.*?SSID: (?P<ssid>.*?)\n").dot_matches_new_line(true).build().unwrap()
    });
    let caps = RE.captures_iter(lines);
    let wifis: Vec<Wifi> = caps.map(|x| Wifi {x["mac"], x["ssid"], x["channel"], x["signal"], x["seen"]}).collect();
}

fn scan(interface: &str, freqs: Vec<u32>) {
    let cmd = vec!["dev", &interface, "scan", "freq"];
    let freq_list: Vec<String> = freqs.iter().map(|x| x.to_string()).collect();
    let freq_list: Vec<&str> = freq_list.iter().map(|x| x.as_ref()).collect();
    let cmd = [cmd, freq_list].concat();
    let scan_out = Command::new("./iw").args(cmd).output().unwrap();
    parse_scan(String::from_utf8_lossy(&scan_out.stdout).as_ref());
}

fn main() {
    scan("wlan0", vec![2412, 2437, 2462]);
}
