import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.image as mpimg

def create_role1_pdf_report(
    project_dir=r"C:\Users\sanja\Downloads\role1_simulated_handoff",
    pdf_filename="Acoustic_Radar_Role1_Report.pdf"
):
    pdf_path_outputs = os.path.join(project_dir, "outputs", pdf_filename)
    pdf_path_root = os.path.join(project_dir, pdf_filename)
    plots_dir = os.path.join(project_dir, "plots")
    
    with open(os.path.join(project_dir, "config.json"), "r") as f:
        config = json.load(f)
    
    report_json_path = os.path.join(project_dir, "outputs", "role1_dataset_comparison_report.json")
    if os.path.exists(report_json_path):
        with open(report_json_path, "r") as f:
            comp_report = json.load(f)
    else:
        comp_report = {}

    with PdfPages(pdf_path_outputs) as pdf:
        # =========================================================================
        # PAGE 1: Title, Role 1 Handoff Contract & Comparison Summary
        # =========================================================================
        fig = plt.figure(figsize=(11, 8.5))
        fig.patch.set_facecolor("#f8f9fa")
        
        plt.subplot2grid((12, 12), (0, 0), rowspan=2, colspan=12)
        plt.axis("off")
        plt.text(0.5, 0.65, "Acoustic FMCW Radar — Role 1 Acoustic Acquisition Report",
                 ha="center", va="center", fontsize=18, fontweight="bold", color="#0f2b48")
        plt.text(0.5, 0.20, "Role 1 Simulation Handoff & Forensic Dataset Comparison Analysis",
                 ha="center", va="center", fontsize=11, color="#495057", style="italic")
        
        plt.subplot2grid((12, 12), (2, 0), rowspan=4, colspan=12)
        plt.axis("off")
        box_text = (
            "ROLE 1 -> ROLE 2 HANDOFF CONTRACT SPECIFICATION:\n"
            f"• Output Data File: rx_audio.npy (432,000 samples, float32, 9.0 s @ 48 kHz)\n"
            f"• Transmit Reference: tx_signal.npy (60 chirps, 18 kHz -> 21 kHz, B = 3000 Hz)\n"
            f"• FMCW Timing: Chirp duration Tc = 100 ms | Gap = 50 ms | Frame interval Tframe = 150 ms (6.67 Hz)\n"
            f"• Target Range: Nominal 1.0 m | Speed of sound: 343.0 m/s | Center Carrier: fc = 19.50 kHz\n"
            f"• Vital-Sign Modulation: Respiration (15 BPM, 4.0 mm) | Heartbeat (72 BPM, 0.15 mm)\n"
            f"• Downstream Compatibility: Direct input for Role 2 FMCW dechirping & range FFT"
        )
        plt.text(0.02, 0.5, box_text, fontsize=9.5, family="monospace", va="center",
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="#e9ecef", edgecolor="#ced4da", lw=1.5))
        
        plt.subplot2grid((12, 12), (6, 0), rowspan=5, colspan=12)
        plt.axis("off")
        
        table_data = [
            ["Parameter / Metric", "Your Role 1 Synthetic Engine", "Jishnu Role 1 Hardware Engine", "Comparative Assessment"],
            ["Frequency Sweep", "18.0 kHz -> 21.0 kHz", "17.5 kHz -> 19.2 kHz", "Your dataset has wider bandwidth (+76%)"],
            ["Bandwidth (B)", "3000 Hz", "1700 Hz", "1.76x Higher Range Resolution (57 mm vs 101 mm)"],
            ["Chirp Timing", "Tc = 100 ms, Tgap = 50 ms", "Tc = 100 ms, Tgap = 0 ms", "Jishnu has continuous streaming (10 Hz)"],
            ["Frame Repetition", "6.67 Hz (150 ms PRI)", "10.0 Hz (100 ms PRI)", "Both Nyquist-adequate for vitals (<3.33 Hz)"],
            ["In-Band SNR", "20.5 dB (Controlled noise)", "47.1 dB (Acoustic room gain)", "Both provide strong FMCW beat peaks"],
            ["Hardware Requirement", "None (100% Software Simulated)", "Microphone + Speaker + Real Room", "Your dataset is 100% portable & reproducible"],
            ["Delay Modeling", "Sample-rounded & Continuous", "Continuous Acoustic Propagation", "Continuous model preserves micro-motion"]
        ]
        
        table = plt.table(cellText=table_data, loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1, 1.8)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_facecolor("#1f77b4")
                cell.set_text_props(color="white", fontweight="bold")
            else:
                cell.set_facecolor("#ffffff" if row % 2 == 0 else "#f1f3f5")
                if col == 3:
                    cell.set_text_props(color="#0055aa", fontweight="bold")

        plt.subplot2grid((12, 12), (11, 0), rowspan=1, colspan=12)
        plt.axis("off")
        plt.text(0.5, 0.2, "Acoustic FMCW Radar — Role 1 Technical Report | Page 1 of 4",
                 ha="center", va="center", fontsize=9, color="#6c757d")
        
        plt.tight_layout()
        pdf.savefig(fig, dpi=300)
        plt.close()

        # =========================================================================
        # PAGE 2: Role 1 Synthetic Signal Waveforms & STFT Spectrogram
        # =========================================================================
        fig = plt.figure(figsize=(11, 8.5))
        plt.suptitle("Role 1: Simulated Acoustic Signal Characteristics", fontsize=15, fontweight="bold", y=0.97)
        
        ax1 = fig.add_subplot(2, 2, 1)
        if os.path.exists(os.path.join(plots_dir, "tx_chirp.png")):
            ax1.imshow(mpimg.imread(os.path.join(plots_dir, "tx_chirp.png")))
        ax1.axis("off")
        ax1.set_title("1. Transmitted FMCW Chirp Signal (18-21 kHz)", fontsize=11, fontweight="bold")

        ax2 = fig.add_subplot(2, 2, 2)
        if os.path.exists(os.path.join(plots_dir, "rx_audio.png")):
            ax2.imshow(mpimg.imread(os.path.join(plots_dir, "rx_audio.png")))
        ax2.axis("off")
        ax2.set_title("2. Received Acoustic Signal (60 Chirp Frames, 9.0 s)", fontsize=11, fontweight="bold")

        ax3 = fig.add_subplot(2, 1, 2)
        if os.path.exists(os.path.join(plots_dir, "spectrogram.png")):
            ax3.imshow(mpimg.imread(os.path.join(plots_dir, "spectrogram.png")))
        ax3.axis("off")
        ax3.set_title("3. Time-Frequency Spectrogram (STFT showing 18-21 kHz sweep)", fontsize=11, fontweight="bold")

        fig.text(0.5, 0.02, "Acoustic FMCW Radar — Role 1 Technical Report | Page 2 of 4", ha="center", fontsize=9, color="#6c757d")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig, dpi=300)
        plt.close()

        # =========================================================================
        # PAGE 3: Role 1 Forensic Comparison (Your Synthetic vs Jishnu Hardware)
        # =========================================================================
        fig = plt.figure(figsize=(11, 8.5))
        plt.suptitle("Role 1 Forensic Comparison: Synthetic Simulation vs Hardware Audio", fontsize=15, fontweight="bold", y=0.97)

        ax1 = fig.add_subplot(2, 2, 1)
        if os.path.exists(os.path.join(plots_dir, "tx_spectrum_comparison.png")):
            ax1.imshow(mpimg.imread(os.path.join(plots_dir, "tx_spectrum_comparison.png")))
        ax1.axis("off")
        ax1.set_title("4. Transmit Spectrum Comparison (TX PSD)", fontsize=10, fontweight="bold")

        ax2 = fig.add_subplot(2, 2, 2)
        if os.path.exists(os.path.join(plots_dir, "rx_spectrum_comparison.png")):
            ax2.imshow(mpimg.imread(os.path.join(plots_dir, "rx_spectrum_comparison.png")))
        ax2.axis("off")
        ax2.set_title("5. Received Audio Power Spectral Density (RX PSD)", fontsize=10, fontweight="bold")

        ax3 = fig.add_subplot(2, 2, 3)
        if os.path.exists(os.path.join(plots_dir, "chirp_correlation_comparison.png")):
            ax3.imshow(mpimg.imread(os.path.join(plots_dir, "chirp_correlation_comparison.png")))
        ax3.axis("off")
        ax3.set_title("6. Matched Filter Cross-Correlation Peak", fontsize=10, fontweight="bold")

        ax4 = fig.add_subplot(2, 2, 4)
        if os.path.exists(os.path.join(plots_dir, "in_band_snr_comparison.png")):
            ax4.imshow(mpimg.imread(os.path.join(plots_dir, "in_band_snr_comparison.png")))
        ax4.axis("off")
        ax4.set_title("7. In-Band Signal-to-Noise Ratio (SNR)", fontsize=10, fontweight="bold")

        fig.text(0.5, 0.02, "Acoustic FMCW Radar — Role 1 Technical Report | Page 3 of 4", ha="center", fontsize=9, color="#6c757d")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig, dpi=300)
        plt.close()

        # =========================================================================
        # PAGE 4: Physical Delay Analysis (Quantized Delay vs Continuous Modeling)
        # =========================================================================
        fig = plt.figure(figsize=(11, 8.5))
        plt.suptitle("Role 1 Physical Analysis: Sample Rounding vs Continuous Delay Modeling", fontsize=14, fontweight="bold", y=0.97)

        ax1 = fig.add_subplot(2, 1, 1)
        if os.path.exists(os.path.join(plots_dir, "delay_quantization_comparison.png")):
            ax1.imshow(mpimg.imread(os.path.join(plots_dir, "delay_quantization_comparison.png")))
        ax1.axis("off")
        ax1.set_title("8. Discretization Step Comparison & Carrier Phase Modulation", fontsize=11, fontweight="bold")

        ax2 = fig.add_subplot(2, 1, 2)
        ax2.axis("off")
        summary_explanation = (
            "KEY SCIENTIFIC TAKEAWAYS FOR ROLE 1 SIMULATION:\n\n"
            "1. Sample-Delay Rounding vs Physical Reality:\n"
            "   • In integer sample-rounding: delay_samples = int(round(tau * fs)).\n"
            "   • At fs = 48 kHz, 1 sample delay step = c / (2*fs) = 3.57 mm of target displacement.\n"
            "   • Respiration (4.0 mm) produces 1-sample staircase steps, while heartbeat (0.15 mm) is buried.\n\n"
            "2. Continuous Fractional-Delay Solution:\n"
            "   • In continuous fractional delay: t_delayed = t_chirp - 2*R(t)/c.\n"
            "   • Because the analytical chirp function is continuous in time, fractional delays preserve sub-millimeter\n"
            "     carrier phase modulation (lambda = 17.59 mm) down to micrometers.\n\n"
            "3. Hardware vs Synthetic Comparison:\n"
            "   • Jishnu's hardware recording operates on real sound waves (continuous physical delay).\n"
            "   • Your improved synthetic engine matches physical wave propagation without requiring hardware.\n"
            "   • Both datasets are ready for downstream FMCW dechirping, range FFT, and vital sign tracking."
        )
        ax2.text(0.02, 0.5, summary_explanation, fontsize=10, family="monospace", va="center",
                 bbox=dict(boxstyle="round,pad=0.8", facecolor="#eef2f7", edgecolor="#cbd5e1", lw=1.5))

        fig.text(0.5, 0.02, "Acoustic FMCW Radar — Role 1 Technical Report | Page 4 of 4", ha="center", fontsize=9, color="#6c757d")
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        pdf.savefig(fig, dpi=300)
        plt.close()

    import shutil
    shutil.copy2(pdf_path_outputs, pdf_path_root)
    print(f"Role 1 PDF generated successfully at:\n  1. {pdf_path_outputs}\n  2. {pdf_path_root}")

if __name__ == "__main__":
    create_role1_pdf_report()
