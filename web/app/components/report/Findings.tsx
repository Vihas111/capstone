import { Finding, FindingChannel, PredictResponse } from "../../lib/api";
import { FadeUp } from "../motion/FadeUp";
import { SectionLabel } from "../ui/SectionLabel";
import styles from "./Report.module.css";

// Ordered by evidential authority: DrugBank's own words first, then mechanism
// inferred from its annotations, then class-level additive effects.
const CHANNELS: {
  id: FindingChannel;
  title: string;
  blurb: string;
}[] = [
  {
    id: "documented_outcome",
    title: "Stated by DrugBank",
    blurb: "Outcomes named directly in DrugBank's own interaction text.",
  },
  {
    id: "shared_protein",
    title: "Shared mechanism",
    blurb:
      "Drugs acting on the same enzyme or transporter, so one drug changes another's exposure.",
  },
  {
    id: "risk_axis",
    title: "Additive effects",
    blurb:
      "Separate mechanisms pushing the same direction. These add up without any shared protein or documented interaction.",
  },
];

export function Findings({ result }: { result: PredictResponse }) {
  const findings = result.findings ?? [];
  const coverage = result.coverage;
  const notes = coverage?.notes ?? [];

  if (!findings.length && !notes.length) return null;

  const byChannel = (id: FindingChannel): Finding[] =>
    findings.filter((f) => f.channel === id);

  return (
    <FadeUp className={styles.section}>
      <SectionLabel>Findings</SectionLabel>

      {!findings.length && (
        <p className={styles.convergenceIntro}>
          No interaction findings from any channel.
        </p>
      )}

      {CHANNELS.map(({ id, title, blurb }) => {
        const group = byChannel(id);
        if (!group.length) return null;
        return (
          <div key={id} className={styles.channelGroup}>
            <div className={styles.channelTitle}>{title}</div>
            <div className={styles.channelBlurb}>{blurb}</div>
            {group.map((f, i) => (
              <div key={i} className={styles.convergenceCard}>
                <div className={styles.convergenceFinding}>{f.finding}</div>
              </div>
            ))}
          </div>
        );
      })}

      {/* Coverage is what stops silence reading as safety: a drug we had no
          data for is otherwise indistinguishable from one that came back clean. */}
      {notes.length > 0 && (
        <div className={styles.coverageBox}>
          <div className={styles.coverageTitle}>Limits of this check</div>
          {notes.map((n, i) => (
            <div key={i} className={styles.coverageNote}>
              {n}
            </div>
          ))}
        </div>
      )}
    </FadeUp>
  );
}
