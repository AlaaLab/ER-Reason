import pandas as pd

class ERReasonDataset:
    def __init__(self, path_to_csv):
        """
        Load the ER Reason dataset CSV into a pandas DataFrame.

        Args:
            path_to_csv (str): Path to the CSV file.
        """
        self.df = pd.read_csv(path_to_csv)
    
    def get_encounter_data(self, encounterkey):
        """
        Return all rows matching a given encounter_key.

        Args:
            encounterkey (str or int): Encounter identifier.

        Returns:
            pd.DataFrame: Rows for that encounter.
        """
        return self.df[self.df['encounterkey'] == encounterkey]

    def get_patient_data(self, patientdurablekey):
        """
        Return all rows matching a given patientdurablekey.

        Args:
            patientdurablekey (str or int): Patient identifier.

        Returns:
            pd.DataFrame: Rows for that patient.
        """
        return self.df[self.df['patientdurablekey'] == patientdurablekey]

    def get_all_patients(self):
        """
        Get unique patient IDs in the dataset.

        Returns:
            np.ndarray: Unique patientdurablekey values.
        """
        return self.df['patientdurablekey'].unique()

    def iter_encounters(self):
        """
        Iterate over all unique encounters.

        Yields:
            Tuple (encounterkey, pd.DataFrame): Encounter key and data for that encounter.
        """
        for ek in self.df['encounterkey'].unique():
            yield ek, self.get_encounter_data(ek)

    def iter_patients(self):
        """
        Iterate over all unique patients.

        Yields:
            Tuple (patientdurablekey, pd.DataFrame): Patient key and data for that patient.
        """
        for pk in self.get_all_patients():
            yield pk, self.get_patient_data(pk)

    def _get_note_text(self, encounterkey, column_name):
        """
        Helper function to get text for a given note column and encounter.

        Args:
            encounterkey (str or int): Encounter identifier.
            column_name (str): Column name for the note text.

        Returns:
            str or None: Text if exists, else None.
        """
        encounter_rows = self.get_encounter_data(encounterkey)
        if not encounter_rows.empty and column_name in encounter_rows.columns:
            val = encounter_rows[column_name].values[0]
            if pd.notna(val):
                return val
        return None

    def get_discharge_summary(self, encounterkey):
        return self._get_note_text(encounterkey, 'Discharge_Summary_Text')

    def get_progress_note(self, encounterkey):
        return self._get_note_text(encounterkey, 'Progress_Note_Text')

    def get_history_physical(self, encounterkey):
        return self._get_note_text(encounterkey, 'HP_Note_Text')

    def get_consult_note(self, encounterkey):
        return self._get_note_text(encounterkey, 'Consult_Text')

    def get_imaging_report(self, encounterkey):
        return self._get_note_text(encounterkey, 'Imaging_Text')

    def get_echo_report(self, encounterkey):
        return self._get_note_text(encounterkey, 'Echo_Text')

    def get_er_provider_notes(self, encounterkey):
        return self._get_note_text(encounterkey, 'ED_Provider_Notes_Text')

    def get_ecg_report(self, encounterkey):
        return self._get_note_text(encounterkey, 'ECG_Text')

    def get_columns(self):
        """
        Get all column names of the dataset.

        Returns:
            list: Column names.
        """
        return self.df.columns.tolist()
