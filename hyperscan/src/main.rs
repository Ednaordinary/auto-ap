use std::thread;
use std::time::Duration;

use netlink_rust::{Protocol, Socket, generic};
use netlink_wi::{self, NlSocket};
use nl80211_rs::get_wireless_interfaces;

fn scan_loop() {
    //let mut control = Socket::new(Protocol::Generic).unwrap();
    //let family = generic::Family::from_name(&mut control, "nl80211").unwrap();
    //let devs = get_wireless_interfaces(&mut control, &family).unwrap();
    // assume one scanner <:
    //let dev = devs.iter().nth(0).unwrap();
    //thread::sleep(Duration::from_millis(500));
    //let scan_res = dev.trigger_scan(&mut control);
    //dev.get_survey(&mut control).unwrap();
    //dev.set_channel(&mut control, 1).unwrap();
    //dev.set_channel(&mut control, 6).unwrap();
    //dev.set_channel(&mut control, 11).unwrap();
    let socket = NlSocket::connect().unwrap();
    //for interface in socket.list_interfaces().unwrap() {
    //    println!("Interface {} {}", interface.interface_index, interface.name);
    //}
    let interface = socket
        .list_interfaces()
        .unwrap()
        .into_iter()
        .filter(|x| x.name == "wlan0")
        .nth(0)
        .unwrap();
    //let interface = socket.get_interface(0);
    //println!("{} {}", interface.name, interface.frequency.unwrap());
    // 2412, 2437, 2462
    let config = netlink_wi::ChannelConfig::new(
        interface.interface_index,
        2412,
        netlink_wi::interface::ChannelWidth::Width20,
    );
    socket.set_channel(config).unwrap();
    println!("{} {}", interface.name, interface.frequency.unwrap());
}

fn main() {
    scan_loop();
}
