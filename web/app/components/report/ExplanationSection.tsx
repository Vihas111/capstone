import { PredictResponse } from "../../lib/api";
import { FadeUp } from "../motion/FadeUp";
import { SectionLabel } from "../ui/SectionLabel";
import styles from "./Report.module.css";

export function ExplanationSection({ result }: { result: PredictResponse }) {
  if (!result.explanation && !result.report?.note) return null;

  return (
    <FadeUp className={styles.section}>
      <SectionLabel>What's Happening</SectionLabel>
      {result.explanation && <p className={styles.explanation}>{result.explanation}</p>}
      {result.report?.note && <p className={styles.recommendation}>{result.report.note}</p>}
    </FadeUp>
  );
}
