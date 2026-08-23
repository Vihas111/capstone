"use client";

import { AnimatePresence, motion } from "framer-motion";
import { InvestigationView } from "./components/investigation/InvestigationView";
import { ReportDossier } from "./components/report/ReportDossier";
import { FadeUp } from "./components/motion/FadeUp";
import { AnalyzeButton } from "./components/ui/AnalyzeButton";
import { DrugSearchInput } from "./components/ui/DrugSearchInput";
import { useAnalysisFlow } from "./hooks/useAnalysisFlow";
import styles from "./page.module.css";

export default function Home() {
  const {
    phase,
    drugs,
    drugCount,
    setDrugCount,
    drugA,
    drugB,
    drugC,
    setDrugA,
    setDrugB,
    setDrugC,
    canAnalyze,
    bootstrapError,
    bootstrapLoading,
    investigationDone,
    result,
    error,
    analyze,
    cancel,
    reset,
    retryBootstrap,
    completeInvestigation,
  } = useAnalysisFlow();

  const inputsDisabled = bootstrapLoading || Boolean(bootstrapError);

  return (
    <main className={styles.workspace}>
      <AnimatePresence mode="wait">
        {phase === "landing" && (
          <FadeUp key="landing" className={styles.landing}>
            <h1 className={styles.title}>Drug Interaction Analysis</h1>
            <div className={styles.toggleRow} role="group" aria-label="Number of drugs">
              <button
                type="button"
                className={`${styles.toggleButton} ${drugCount === 2 ? styles.toggleButtonActive : ""}`}
                onClick={() => setDrugCount(2)}
                disabled={inputsDisabled}
              >
                2 Drugs
              </button>
              <button
                type="button"
                className={`${styles.toggleButton} ${drugCount === 3 ? styles.toggleButtonActive : ""}`}
                onClick={() => setDrugCount(3)}
                disabled={inputsDisabled}
              >
                3 Drugs
              </button>
            </div>
            <div className={`${styles.inputs} ${drugCount === 3 ? styles.inputsThree : ""}`}>
              <DrugSearchInput
                id="drug-a"
                label="Drug A"
                value={drugA}
                onChange={setDrugA}
                drugs={drugs}
                disabled={inputsDisabled}
              />
              <div className={styles.cross}>×</div>
              <DrugSearchInput
                id="drug-b"
                label="Drug B"
                value={drugB}
                onChange={setDrugB}
                drugs={drugs}
                disabled={inputsDisabled}
              />
              {drugCount === 3 && (
                <>
                  <div className={styles.cross}>×</div>
                  <DrugSearchInput
                    id="drug-c"
                    label="Drug C"
                    value={drugC}
                    onChange={setDrugC}
                    drugs={drugs}
                    disabled={inputsDisabled}
                  />
                </>
              )}
            </div>
            {bootstrapLoading && (
              <p className={styles.status} role="status">
                Loading drug records…
              </p>
            )}
            {bootstrapError && (
              <div className={styles.bootstrapError} role="alert">
                <p className={styles.status}>{bootstrapError}</p>
                <button
                  type="button"
                  className={styles.retryAction}
                  onClick={() => void retryBootstrap()}
                >
                  Retry loading drug records
                </button>
              </div>
            )}
            <motion.div
              className={styles.actions}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: canAnalyze ? 1 : 0.35, y: 0 }}
            >
              <AnalyzeButton disabled={!canAnalyze} onClick={analyze} />
            </motion.div>
          </FadeUp>
        )}
        {phase === "investigating" && (
          <InvestigationView
            key="investigating"
            started
            done={investigationDone}
            onComplete={completeInvestigation}
            onCancel={cancel}
          />
        )}
        {phase === "report" && result && (
          <ReportDossier key="report" result={result} onReset={reset} />
        )}
        {phase === "error" && (
          <FadeUp key="error" className={styles.errorView}>
            <h2 className={styles.errorTitle}>Analysis stopped</h2>
            <p className={styles.errorMessage}>{error ?? "Connection failed."}</p>
            <button type="button" className={styles.errorAction} onClick={reset}>
              Start over
            </button>
          </FadeUp>
        )}
      </AnimatePresence>
    </main>
  );
}
