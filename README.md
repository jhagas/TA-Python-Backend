# Python Backend for my Final Project

## For WSL 2:

1. Install USBIPD-WIN:
   Download and install the usbipd-win tool, which is required to forward USB devices to WSL 2.

```powershell
winget install --interactive --exact dorssel.usbipd-win
```

3. List USB Devices:

- List the available USB devices to identify your Arduino:

```powershell
usbipd list
```

4. Attach USB Device to WSL:
   Use the device ID from the previous step to attach the USB device to your WSL instance:

```powershell
bind --busid <busid>
usbipd attach --wsl --busid <busid>
```

Replace `<busid>` with the actual bus ID of your Arduino.

5. Access USB Device in WSL:
   Now, you can access the USB device from within your WSL environment. It will typically show up as /dev/ttyUSB0 or similar.
