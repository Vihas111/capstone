import { PredictResponse } from "../../lib/api";
import { FadeUp } from "../motion/FadeUp";
import { AdverseEventsRank } from "./AdverseEventsRank";
import { EvidenceAccordion } from "./EvidenceAccordion";
import { ExplanationSection } from "./ExplanationSection";
import { InteractionHero } from "./InteractionHero";
import { Findings } from "./Findings";
import styles from "./Report.module.css";

interface ReportDossierProps {
  result: PredictResponse;
  onReset: () => void;
}

export function ReportDossier({ result, onReset }: ReportDossierProps) {
  return (
    <FadeUp className={styles.report}>
      <InteractionHero result={result} />
      <ExplanationSection result={result} />
      <Findings result={result} />
      <EvidenceAccordion result={result} />
      <AdverseEventsRank result={result} />
      <button type="button" onClick={onReset} className={styles.newAnalysis}>
        Start new analysis
      </button>
    </FadeUp>
  );
}
