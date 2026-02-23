using System;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Windows.Forms;

namespace EzFramesLauncher
{
    internal static class Program
    {
        [STAThread]
        private static int Main(string[] args)
        {
            string exePath = Process.GetCurrentProcess().MainModule?.FileName ?? Application.ExecutablePath;
            string installRoot = Path.GetDirectoryName(exePath) ?? AppDomain.CurrentDomain.BaseDirectory;

            string pythonw = Path.Combine(installRoot, "runtime", "python", "pythonw.exe");
            string bootstrap = Path.Combine(installRoot, "bootstrap_launcher.py");

            if (!File.Exists(pythonw) || !File.Exists(bootstrap))
            {
                MessageBox.Show(
                    "EzFrames launcher files are missing.\n\nPlease reinstall EzFrames.",
                    "EzFrames",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                return 2;
            }

            string forwardedArgs = string.Join(
                " ",
                args.Select(EscapeArg));
            string bootstrapArg = EscapeArg(bootstrap);
            string launchArgs = string.IsNullOrWhiteSpace(forwardedArgs)
                ? bootstrapArg
                : $"{bootstrapArg} {forwardedArgs}";

            var startInfo = new ProcessStartInfo
            {
                FileName = pythonw,
                Arguments = launchArgs,
                WorkingDirectory = installRoot,
                UseShellExecute = false,
                CreateNoWindow = true,
            };

            try
            {
                Process.Start(startInfo);
                return 0;
            }
            catch (Exception ex)
            {
                MessageBox.Show(
                    "Failed to start EzFrames.\n\n" + ex.Message,
                    "EzFrames",
                    MessageBoxButtons.OK,
                    MessageBoxIcon.Error);
                return 1;
            }
        }

        private static string EscapeArg(string value)
        {
            if (string.IsNullOrEmpty(value))
            {
                return "\"\"";
            }

            if (!value.Contains(' ') && !value.Contains('"'))
            {
                return value;
            }

            return "\"" + value.Replace("\"", "\\\"") + "\"";
        }
    }
}
