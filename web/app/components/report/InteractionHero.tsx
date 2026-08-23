import { InteractionStatus, PredictResponse } from "../../lib/api";
import { SectionLabel } from "../ui/SectionLabel";
import styles from "./Report.module.css";

// Deliberately NOT a red/amber/green danger scale. These three states describe
// what EVIDENCE exists, not how severe the interaction is -- nothing in this
// pipeline is validated against severity-graded data, so a traffic-light
// palette would imply a judgement the models cannot make.
const STATUS_DISPLAY: Record<
  InteractionStatus,
  { label: string; color: string }
> = {
  documented: { label: "INTERACTION DOCUMENTED", color: "var(--accent)" },
  predicted: { label: "INTERACTION PREDICTED", color: "var(--accent-soft)" },
  none_found: { label: "NO INTERACTION FOUND", color: "var(--muted)" },
};

function subtitle(result: PredictResponse): string | null {
  const report = result.report;
  if (!report || report.pairs_total <= 1) return null;
  const bits = [`${report.pairs_total} pairs checked`];
  if (report.pairs_documented) bits.push(`${report.pairs_documented} documented`);
  if (report.pairs_predicted) bits.push(`${report.pairs_predicted} predicted only`);
  return bits.join(" · ");
}

export function InteractionHero({ result }: { result: PredictResponse }) {
  const status = result.report?.interaction_status ?? "none_found";
  const display = STATUS_DISPLAY[status] ?? STATUS_DISPLAY.none_found;
  const names = result.drugs ?? [];
  const ids = result.drug_ids ?? [];
  const sub = subtitle(result);

  return (
    <section>
      <div className={styles.heroRow}>
        {names.map((name, index) => (
          <span key={`${name}-${index}`} className={styles.heroRow}>
            <h1 className={styles.heroNames}>{name}</h1>
            {index < names.length - 1 && (
              <span className={styles.cross} aria-hidden>
                ×
              </span>
            )}
          </span>
        ))}
      </div>

      <div className={styles.riskRow}>
        <span className={styles.riskAccent} style={{ background: display.color }} />
        <div>
          <SectionLabel>Result</SectionLabel>
          <p className={styles.riskValue}>{display.label}</p>
          {sub && <p className={styles.statusSub}>{sub}</p>}
        </div>
      </div>

      <div className={styles.meta}>{ids.filter(Boolean).join(" · ")}</div>
    </section>
  );
}
